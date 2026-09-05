from __future__ import annotations

import copy
import io
import math
from dataclasses import asdict, replace
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch
from tokenizers import Tokenizer, models, processors
from torch import nn

from metis_ablation.evaluate import (
    build_parser,
    json_sha256,
    load_checkpoint,
    load_model,
    special_token_id,
    validate_completed_checkpoint,
    validate_coverage,
    validate_prepared_requests,
)
from metis_ablation.lm_eval_adapter import (
    ContextOverflowError,
    NativeDenseLM,
    TokenBoundaryError,
    as_harness_lm,
    fixed_dense_curriculum,
    rolling_windows,
)
from metis_training.model import CurriculumState, Metis16ForCausalLM
from metis_training.model_config import Metis16Config


class CharacterTokenizer:
    truncation = None
    padding = None

    def __init__(self):
        self.vocabulary = {"<eos>": 0, "<pad>": 1, "<bos>": 2}
        self.vocabulary.update({char: index + 3 for index, char in enumerate(" abcxyzQ!")})
        self.reverse = {index: text for text, index in self.vocabulary.items()}

    def get_vocab_size(self, with_added_tokens=True):
        return len(self.vocabulary)

    def token_to_id(self, token):
        return self.vocabulary.get(token)

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("Postprocessor must not insert hidden BOS/EOS tokens")
        return SimpleNamespace(ids=[self.vocabulary[char] for char in text])

    def decode(self, tokens, *, skip_special_tokens):
        if skip_special_tokens:
            raise AssertionError("Only explicit EOS/text stops may remove generated content")
        return "".join(self.reverse[token] for token in tokens)


def tiny_config(vocab_size: int, *, max_length: int = 32) -> Metis16Config:
    return replace(
        Metis16Config.tiny_for_tests(),
        n_layers=2, vocab_size=vocab_size, ffn_mode="dense",
        n_routed_experts=0, n_shared_experts=0, dense_ffn_intermediate_dim=24,
        final_context_length=max_length, max_passes=5,
    )


def saved_curriculum() -> CurriculumState:
    return CurriculumState(
        continuation_mode="depth_one", memory_gate_scale=0.0,
        ngram_gate_scale=1.0, stochastic_routing=True, target_mean_depth=2.0,
    )


class TransitionModel(nn.Module):
    """Exact bigram oracle whose logits and call geometry are observable."""

    def __init__(self, tokenizer, *, max_length=32, transitions=None):
        super().__init__()
        vocab = tokenizer.get_vocab_size()
        self.config = tiny_config(vocab, max_length=max_length)
        self.embedding = nn.Embedding(vocab, vocab)
        self.lm_head = nn.Identity()
        self.calls = []
        self.bad_depth = False
        with torch.no_grad():
            if transitions is None:
                self.embedding.weight.copy_(
                    torch.arange(vocab * vocab).reshape(vocab, vocab).remainder(17).float() / 3,
                )
            else:
                self.embedding.weight.fill_(-8)
                self.embedding.weight[:, 0] = 8
                for previous, following in transitions.items():
                    row = tokenizer.token_to_id(previous)
                    target = tokenizer.token_to_id(following)
                    self.embedding.weight[row].fill_(-8)
                    self.embedding.weight[row, target] = 8

    def _precision_call(self, function, hidden):
        return function(hidden)

    def forward(self, ids, **kwargs):
        self.calls.append({
            "ids": ids.detach().clone(), "training": self.training,
            "grad_enabled": torch.is_grad_enabled(),
            **{key: value.detach().clone() if isinstance(value, torch.Tensor) else value
               for key, value in kwargs.items()},
        })
        mask = kwargs["attention_mask"]
        return SimpleNamespace(
            final_hidden_state=self.embedding(ids),
            chosen_depths=mask.long() * (2 if self.bad_depth else 1),
        )


def request(*args):
    return SimpleNamespace(args=args)


def adapter(tokenizer=None, *, model=None, batch_size=2, **kwargs):
    tokenizer = tokenizer or CharacterTokenizer()
    model = model or TransitionModel(tokenizer)
    return NativeDenseLM(
        model, tokenizer, curriculum=saved_curriculum(),
        canonical_id_lookup=list(reversed(range(tokenizer.get_vocab_size()))),
        eos_token_id=0, pad_token_id=1, batch_size=batch_size,
        logit_chunk_size=1, **kwargs,
    )


class TokenizationTests(unittest.TestCase):
    def test_trailing_prompt_whitespace_belongs_to_continuation(self):
        lm = adapter()
        context, targets = lm._encode_pair("a ", "bc")
        self.assertEqual(context, lm.tok_encode("a"))
        self.assertEqual(targets, lm.tok_encode(" bc"))
        whitespace_context, whitespace_target = lm._encode_pair(" ", "a")
        self.assertEqual(whitespace_context, [0])
        self.assertEqual(whitespace_target, lm.tok_encode(" a"))

    def test_joint_bpe_boundary_change_is_not_silently_dropped_or_rescored(self):
        vocab = {"<eos>": 0, "<pad>": 1, "<bos>": 2, "a": 3, "b": 4, "ab": 5, " ": 6}
        tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[("a", "b")]))
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<bos> $A <eos>", special_tokens=[("<bos>", 2), ("<eos>", 0)],
        )
        lm = adapter(tokenizer)
        self.assertEqual(lm.tok_encode("ab"), [5])
        self.assertEqual(lm._encode_pair("", "ab"), ([0], [5]))
        self.assertEqual(lm._encode_pair("a ", "b"), ([3], [6, 4]))
        with self.assertRaisesRegex(TokenBoundaryError, "not token-aligned"):
            lm.loglikelihood([request("a", "b")])
        self.assertEqual(lm.model.calls, [])

    def test_bos_is_explicit_and_never_scored_as_a_target(self):
        default = adapter()
        with_bos = adapter(bos_token_id=2)
        self.assertEqual(default._encode_pair("a", "b"), (default.tok_encode("a"), default.tok_encode("b")))
        self.assertEqual(with_bos._encode_pair("a", "b"), ([2, *with_bos.tok_encode("a")], with_bos.tok_encode("b")))
        self.assertEqual(with_bos._encode_pair("", "b"), ([2], with_bos.tok_encode("b")))

    def test_tokenizer_padding_truncation_and_vocabulary_drift_are_rejected(self):
        for name in ("truncation", "padding"):
            tokenizer = CharacterTokenizer()
            setattr(tokenizer, name, {"enabled": True})
            with self.subTest(name=name), self.assertRaises(ValueError):
                adapter(tokenizer)
        tokenizer = CharacterTokenizer()
        model = TransitionModel(tokenizer)
        model.config = replace(model.config, vocab_size=model.config.vocab_size + 1)
        with self.assertRaisesRegex(ValueError, "vocabulary sizes differ"):
            adapter(tokenizer, model=model)


class LikelihoodTests(unittest.TestCase):
    def expected(self, lm, previous, target):
        logits = lm.model.embedding.weight[previous].detach().tolist()
        return logits[target] - math.log(sum(math.exp(value) for value in logits))

    def test_exact_conditional_probability_and_greedy_flag(self):
        tokenizer = CharacterTokenizer()
        model = TransitionModel(tokenizer, transitions={"a": "b", "b": "c"})
        lm = adapter(tokenizer, model=model)
        a, b, c, x = (tokenizer.token_to_id(char) for char in "abcx")
        good, bad = lm.loglikelihood([request("a", "bc"), request("a", "bx")])
        self.assertAlmostEqual(good[0], self.expected(lm, a, b) + self.expected(lm, b, c), places=6)
        self.assertAlmostEqual(bad[0], self.expected(lm, a, b) + self.expected(lm, b, x), delta=2e-6)
        self.assertTrue(good[1])
        self.assertFalse(bad[1])
        self.assertLess(bad[0], good[0])

    def test_only_continuation_tokens_scored_not_prompt_padding_or_eos(self):
        lm = adapter()
        ids = lm.tok_encode("abcxy")
        score, _ = lm.loglikelihood([request("abc", "xy")])[0]
        self.assertAlmostEqual(score, self.expected(lm, ids[2], ids[3]) + self.expected(lm, ids[3], ids[4]), places=5)
        self.assertEqual(lm.stats["scored_tokens"], 2)
        self.assertEqual(lm.model.calls[0]["ids"].tolist(), [ids[:-1]])
        self.assertEqual(lm.loglikelihood([request("anything", "")]), [(0.0, True)])

    def test_batching_padding_permutation_and_canonical_ids(self):
        requests = [request("abc", "xy"), request("a", "b"), request("", "c")]
        single = adapter(batch_size=1)
        batched = adapter(batch_size=3)
        expected = single.loglikelihood(requests)
        actual = batched.loglikelihood(requests)
        self.assertEqual(actual, expected)
        self.assertEqual(batched.loglikelihood(list(reversed(requests))), list(reversed(expected)))
        call = batched.model.calls[0]
        self.assertEqual(call["ids"].shape, (3, 4))
        self.assertEqual(call["attention_mask"].tolist(), [[True] * 4, [True, False, False, False], [True, False, False, False]])
        self.assertEqual(call["canonical_ids"].tolist(),
                         (batched.config.vocab_size - 1 - call["ids"]).tolist())
        self.assertTrue(bool(call["reset_mask"][:, 0].all()))
        self.assertTrue(bool(call["document_ids"][~call["attention_mask"]].eq(-1).all()))
        self.assertFalse(call["training"])
        self.assertFalse(call["grad_enabled"])
        self.assertEqual(call["max_passes"], 1)
        self.assertEqual(call["force_depth"], 1)
        self.assertFalse(call["return_logits"])
        self.assertEqual(call["curriculum"].continuation_mode, "depth_one")
        self.assertEqual(call["curriculum"].memory_gate_scale, 0.0)
        self.assertEqual(call["curriculum"].ngram_gate_scale, 1.0)
        self.assertFalse(call["curriculum"].stochastic_routing)

    def test_context_overflow_preflights_entire_request_batch(self):
        lm = adapter(max_length=3)
        with self.assertRaises(ContextOverflowError):
            lm.loglikelihood([request("a", "b"), request("abc", "xy")])
        self.assertEqual(lm.model.calls, [])
        self.assertEqual(len(lm.loglikelihood([request("abc", "x")])), 1)

    def test_exact_context_reuse_preserves_choices_order_and_duplicate_requests(self):
        requests = [request("abc", "x"), request("abc", "y"), request("a", "b"),
                    request("", "y"), request("abc", "x")]
        baseline = adapter(batch_size=3, reuse_single_token_contexts=False)
        reused = adapter(batch_size=3)
        self.assertEqual(reused.loglikelihood(requests), baseline.loglikelihood(requests))
        self.assertLess(reused.stats["forward_calls"], baseline.stats["forward_calls"])
        self.assertEqual(reused.stats["single_token_context_reuse_hits"], 2)
        self.assertEqual(reused.stats["completed_likelihood_pairs"], len(requests))

    def test_bad_depth_or_nonfinite_logits_fail_closed(self):
        lm = adapter()
        lm.model.bad_depth = True
        with self.assertRaisesRegex(RuntimeError, "exactly one pass"):
            lm.loglikelihood([request("a", "b")])
        lm.model.bad_depth = False
        with torch.no_grad():
            lm.model.embedding.weight.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            lm.loglikelihood([request("a", "b")])

    def test_rolling_covers_each_target_once_with_full_final_context(self):
        self.assertEqual(
            list(rolling_windows(list(range(10)), max_length=4, prefix_token_id=99)),
            [([99], [0, 1, 2, 3]), ([3], [4, 5, 6, 7]), ([5, 6, 7], [8, 9])],
        )
        for maximum in range(1, 7):
            for length in range(4 * maximum + 2):
                tokens = list(range(length))
                windows = list(rolling_windows(tokens, max_length=maximum, prefix_token_id=99))
                self.assertEqual([token for _context, target in windows for token in target], tokens)
                for index, (context, target) in enumerate(windows):
                    self.assertTrue(context)
                    self.assertTrue(target)
                    self.assertLessEqual(len(context) + len(target) - 1, maximum)
                    if index:
                        self.assertEqual(len(context) + len(target) - 1, maximum)

    def test_rolling_probabilities_keep_documents_independent(self):
        lm = adapter(max_length=3)
        texts = ["abcxyzabcx", "bca", ""]
        actual = lm.loglikelihood_rolling([request(text) for text in texts])
        for text, score in zip(texts, actual, strict=True):
            ids = [0, *lm.tok_encode(text)]
            expected = sum(self.expected(lm, previous, target) for previous, target in zip(ids, ids[1:]))
            self.assertAlmostEqual(score, expected, places=5)
        self.assertEqual(lm.stats["scored_tokens"], sum(map(len, texts)))
        self.assertEqual(actual[-1], 0.0)


class GenerationTests(unittest.TestCase):
    def lm(self, **kwargs):
        tokenizer = CharacterTokenizer()
        model = TransitionModel(
            tokenizer, transitions={"a": "x", "x": "y", "y": "<eos>", "b": "z", "z": "<eos>"},
        )
        return adapter(tokenizer, model=model, **kwargs)

    def test_text_stops_across_tokens_eos_and_row_compaction(self):
        requests = [
            request("a", {"until": ["xy"], "max_gen_toks": 5}),
            request("a", {"until": ["y"], "max_gen_toks": 5}),
            request("b", {"until": [], "max_gen_toks": 5}),
            request("a", {"until": ["absent"], "max_gen_toks": 5}),
        ]
        single = self.lm(batch_size=1)
        batched = self.lm(batch_size=4)
        self.assertEqual(batched.generate_until(requests), ["", "x", "z", "xy"])
        self.assertEqual(single.generate_until(requests), ["", "x", "z", "xy"])
        self.assertEqual(batched.stats["generation_text_stops"], 2)
        self.assertEqual(batched.stats["generation_eos_stops"], 2)
        self.assertEqual(batched.stats["generation_token_limit_stops"], 0)

    def test_stop_match_is_generated_text_only(self):
        lm = self.lm()
        self.assertEqual(lm.generate_until([request("x", {"until": ["xy"], "max_gen_toks": 4})]), ["y"])

    def test_earliest_match_and_empty_stop(self):
        lm = self.lm()
        self.assertEqual(lm.generate_until([request("a", {"until": ["y", "xy"], "max_gen_toks": 4})]), [""])
        no_forward = self.lm()
        self.assertEqual(no_forward.generate_until([request("a", {"until": [""]})]), [""])
        self.assertEqual(no_forward.generate_until([request("a", {"until": ""})]), [""])
        self.assertEqual(no_forward.model.calls, [])

    def test_greedy_budget_and_kwargs_do_not_mutate_request(self):
        lm = self.lm()
        kwargs = {"until": ["not present"], "max_gen_toks": 1, "temperature": 0.0, "do_sample": False}
        original = copy.deepcopy(kwargs)
        self.assertEqual(lm.generate_until([request("a", kwargs)]), ["x"])
        self.assertEqual(kwargs, original)
        self.assertEqual(lm.stats["generation_token_limit_stops"], 1)
        self.assertEqual(lm.generate_until([request("a", {"max_new_tokens": 2})]), ["xy"])

    def test_official_mmlu_pro_budget_overrides_fallback_without_clipping(self):
        lm = self.lm()
        options = lm._generation_options({
            "until": ["Question:"], "max_gen_toks": 2048,
            "do_sample": False, "temperature": 0.0,
        })
        self.assertEqual(options.max_new_tokens, 2048)
        self.assertEqual(options.until, ("Question:",))
        tokenizer = CharacterTokenizer()
        model = TransitionModel(tokenizer, max_length=300, transitions={"a": "x", "x": "x"})
        lm = adapter(tokenizer, model=model)
        self.assertEqual(lm.max_gen_toks, 256)
        self.assertEqual(lm.generate_until([request("a", {"max_gen_toks": 257})]), ["x" * 257])
        self.assertEqual(lm.stats["generated_tokens"], 257)

    def test_no_silent_sampling_or_unsupported_generation_options(self):
        for kwargs in (
            {"do_sample": True}, {"top_p": 0.9}, {"max_length": 12},
            {"max_gen_toks": 0}, {"max_gen_toks": 2, "max_new_tokens": 2},
            {"temperature": float("nan")}, {"until": [3]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.lm().generate_until([request("a", kwargs)])

    def test_prompt_and_live_prefix_overflow_never_crop_or_return_partial_output(self):
        lm = self.lm(max_length=2)
        with self.assertRaises(ContextOverflowError):
            lm.generate_until([request("abc", {"max_gen_toks": 3})])
        self.assertEqual(lm.model.calls, [])
        with self.assertRaises(ContextOverflowError):
            lm.generate_until([request("a", {"max_gen_toks": 3})])
        self.assertEqual([call["ids"].shape[1] for call in lm.model.calls], [1, 2])
        self.assertEqual(lm.stats["generation_possible_context_overflows"], 1)
        early_stop = self.lm(max_length=2)
        self.assertEqual(early_stop.generate_until([request("a", {"until": ["xy"], "max_gen_toks": 10})]), [""])

    def test_full_prefix_cost_is_observable_and_no_samples_are_logged(self):
        lm = self.lm()
        lm.generate_until([request("Qa", {"max_gen_toks": 3})])
        metadata = lm.metadata()
        self.assertEqual(metadata["stats"]["forward_calls"], 3)
        self.assertEqual(metadata["stats"]["forward_input_tokens"], 2 + 3 + 4)
        self.assertEqual(len(metadata["request_sha256"]["generate_until"]), 64)
        self.assertNotIn("Qa", str(metadata))
        self.assertEqual(metadata["generation_backend"], "full_prefix_no_cache")


class NativeModelParityTests(unittest.TestCase):
    def test_tiny_native_dense_saved_depth_and_selected_head_match_full_logits(self):
        torch.manual_seed(1701)
        tokenizer = CharacterTokenizer()
        config = tiny_config(tokenizer.get_vocab_size())
        model = Metis16ForCausalLM(config, dtype=torch.float32).eval()
        lm = adapter(tokenizer, model=model, batch_size=3)
        requests = [request("abc", "xy"), request("a", "bc"), request("", "y")]
        actual = lm.loglikelihood(requests)
        expected = []
        for item in requests:
            context, target = lm._encode_pair(*item.args)
            ids = torch.tensor([(context + target)[:-1]])
            mask = torch.ones_like(ids, dtype=torch.bool)
            resets = torch.zeros_like(mask)
            resets[:, 0] = True
            with torch.no_grad():
                output = model(
                    ids, curriculum=saved_curriculum(), attention_mask=mask,
                    reset_mask=resets, canonical_ids=lm.canonical_id_lookup[ids],
                    return_logits=True,
                )
            self.assertTrue(bool(output.chosen_depths.eq(1).all()))
            self.assertEqual(output.active_masks.shape[0], config.max_passes)
            selected = output.logits[0, len(context) - 1:].float()
            targets = torch.tensor(target)
            expected.append((
                float(selected.log_softmax(-1).gather(-1, targets[:, None]).sum(dtype=torch.float64)),
                bool(selected.argmax(-1).eq(targets).all()),
            ))
        for observed, reference in zip(actual, expected, strict=True):
            self.assertAlmostEqual(observed[0], reference[0], places=5)
            self.assertEqual(observed[1], reference[1])
        permuted = lm.loglikelihood(list(reversed(requests)))
        for observed, reference in zip(permuted, reversed(expected), strict=True):
            self.assertAlmostEqual(observed[0], reference[0], places=5)

    def test_curriculum_rejects_wrong_control_even_when_config_defaults_look_valid(self):
        config = tiny_config(12)
        good = fixed_dense_curriculum(config, asdict(saved_curriculum()))
        self.assertEqual(config.target_mean_passes, 2.0)
        self.assertEqual(config.max_passes, 5)
        self.assertEqual(good.continuation_mode, "depth_one")
        for state in (
            replace(good, continuation_mode="adaptive"),
            replace(good, memory_gate_scale=1.0),
            replace(good, ngram_gate_scale=0.0),
        ):
            with self.subTest(state=state), self.assertRaises(ValueError):
                fixed_dense_curriculum(config, state)


def checkpoint_fixture():
    identity = {
        "schema": "more.ablation-run-identity/v1",
        "row": "dense-flop-matched", "model": {"ffn_mode": "dense"},
        "curriculum": {"continuation_mode": "depth_one"},
        "sampler": {"total_blocks": 7, "sampled_tokens": 700},
        "schedule": {"total_steps": 7, "base_learning_rate": 0.001},
        "global_batch_tokens": 100, "precision_profile": "fp8",
        "source_revision": "a" * 40,
    }
    digest = json_sha256(identity)
    run = {
        "schema": "more.ablation-run/v1", "run_identity": identity,
        "run_identity_sha256": digest, "spec": {"name": "dense-flop-matched"},
        "total_steps": 7, "final_checkpoint": True,
        **{key: value for key, value in identity.items()
           if key in {"model", "curriculum", "sampler", "schedule", "global_batch_tokens", "precision_profile"}},
    }
    payload = {
        "schema": "more.ablation-checkpoint/v3", "run_identity_sha256": digest,
        "model": {"weight": torch.zeros(1)}, "spec": run["spec"],
        "step": 7, "total_steps": 7, "step_semantics": "next_unexecuted",
        "base_learning_rate": 0.001,
        "optimizer_shards": [{"path": "must-not-be-opened.pt"}],
    }
    return payload, run, digest


class IdentityAndHarnessTests(unittest.TestCase):
    def test_safe_checkpoint_loader_preserves_te_extra_state_without_global_allowlist_changes(self):
        payload, _run, _digest = checkpoint_fixture()
        payload["model"]["projection._extra_state"] = io.BytesIO(b"opaque TE metadata")
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        previous_globals = torch.serialization.get_safe_globals()
        restored = load_checkpoint(buffer, mmap=False)
        self.assertEqual(restored["model"]["projection._extra_state"].getvalue(), b"opaque TE metadata")
        self.assertCountEqual(torch.serialization.get_safe_globals(), previous_globals)
        with torch.serialization.safe_globals([io.BytesIO]):
            buffer.seek(0)
            load_checkpoint(buffer, mmap=False)
            self.assertIn(io.BytesIO, torch.serialization.get_safe_globals())
        with mock.patch("metis_ablation.evaluate.torch.load", return_value=payload) as loader:
            self.assertIs(load_checkpoint("state.pt"), payload)
        loader.assert_called_once_with("state.pt", map_location="cpu", weights_only=True, mmap=True)

    def test_checkpoint_backend_layout_loads_strictly_and_preserves_storage_policy(self):
        model = mock.Mock()
        model.named_parameters.return_value = []
        config = tiny_config(12)
        weights = {"some.weight": torch.ones(1)}
        policy = mock.Mock()
        with (
            mock.patch("metis_ablation.evaluate.torch.cuda.is_available", return_value=True),
            mock.patch("metis_ablation.evaluate.torch.cuda.set_device"),
            mock.patch("metis_ablation.evaluate.build_precision_policy", return_value=policy) as policy_builder,
            mock.patch("metis_ablation.evaluate.Metis16ForCausalLM", return_value=model) as constructor,
        ):
            loaded, loaded_policy = load_model(config, weights, device=torch.device("cuda:0"), checkpoint_precision="fp8")
        self.assertIs(loaded, model)
        self.assertIs(loaded_policy, policy)
        self.assertEqual(policy_builder.call_args.kwargs["profile"], "fp8")
        self.assertFalse(policy_builder.call_args.kwargs["permit_fallback"])
        self.assertEqual(constructor.call_args.kwargs["dtype"], torch.bfloat16)
        model.apply_parameter_storage_policy.assert_called_once_with(device=torch.device("cuda:0"))
        model.load_state_dict.assert_called_once_with(weights, strict=True)
        model.requires_grad_.assert_called_once_with(False)
        model.eval.assert_called_once_with()

    def test_complete_checkpoint_identity_without_opening_optimizer_inventory(self):
        payload, run, digest = checkpoint_fixture()
        with mock.patch("builtins.open", side_effect=AssertionError("No optimizer reads")):
            self.assertEqual(validate_completed_checkpoint(payload, run, expected_run_identity=digest), run["run_identity"])

    def test_incomplete_or_mismatched_checkpoint_fails(self):
        for field, value in (
            ("step", 6), ("total_steps", 8), ("step", True),
            ("step_semantics", "legacy"), ("run_identity_sha256", "f" * 64),
            ("schema", "more.ablation-checkpoint/v2"), ("base_learning_rate", 0.1),
            ("model", {}), ("optimizer", {"state": {}}),
        ):
            payload, run, digest = checkpoint_fixture()
            payload[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_completed_checkpoint(payload, run, expected_run_identity=digest)

    def test_mutated_run_identity_and_unsealed_shadow_fields_fail(self):
        payload, run, digest = checkpoint_fixture()
        run["run_identity"] = copy.deepcopy(run["run_identity"])
        run["run_identity"]["sampler"]["sampled_tokens"] = 699
        with self.assertRaisesRegex(ValueError, "pinned identity"):
            validate_completed_checkpoint(payload, run, expected_run_identity=digest)
        payload, run, digest = checkpoint_fixture()
        run["curriculum"] = {"continuation_mode": "adaptive"}
        with self.assertRaisesRegex(ValueError, "curriculum"):
            validate_completed_checkpoint(payload, run, expected_run_identity=digest)

    def test_full_scores_require_complete_coverage_diagnostics_are_distinct(self):
        result = {"configs": {"task": {}}, "n-samples": {"task": {"original": 10, "effective": 10}}}
        self.assertEqual(validate_coverage(result, diagnostic=False), result["n-samples"])
        result["n-samples"]["task"]["effective"] = 2
        with self.assertRaises(RuntimeError):
            validate_coverage(result, diagnostic=False)
        self.assertEqual(validate_coverage(result, diagnostic=True), result["n-samples"])
        with self.assertRaises(RuntimeError):
            validate_coverage({}, diagnostic=True)

    def test_special_token_ambiguity_is_not_guessed(self):
        tokenizer = CharacterTokenizer()
        self.assertEqual(special_token_id(tokenizer, ("<eos>", "</s>"), required=True), 0)
        self.assertIsNone(special_token_id(tokenizer, ("absent",), required=False))
        with self.assertRaises(ValueError):
            special_token_id(tokenizer, ("<eos>", "<pad>"), required=True)

    def test_diagnostic_flags_and_task_defaults_are_explicit(self):
        args = build_parser().parse_args([
            "--checkpoint", "state.pt", "--run-manifest", "run.json",
            "--expected-run-identity", "0" * 64, "--release-root", "release",
            "--tasks", "mmlu", "arc_easy",
            "--output-dir", "results",
        ])
        self.assertIsNone(args.num_fewshot)
        self.assertIsNone(args.diagnostic_limit)
        self.assertIsNone(args.diagnostic_max_context)
        self.assertIsNone(args.bos_token_id)
        self.assertEqual(args.expected_harness_version, "0.4.13")

    def test_prepared_request_identity_rejects_omission_reordering_and_changed_text(self):
        import hashlib
        import json

        requests = [
            SimpleNamespace(task_name="task", request_type="loglikelihood",
                            doc_id=0, idx=index, arguments=("a", target))
            for index, target in enumerate(("b", "c"))
        ]
        digest = hashlib.sha256()
        for item in requests:
            digest.update(json.dumps(
                ["task", item.doc_id, item.idx, *item.arguments],
                ensure_ascii=False, separators=(",", ":"),
            ).encode() + b"\n")
        expected = {"task": {"requests": 2, "request_sha256": digest.hexdigest()}}
        validate_prepared_requests(requests, expected)
        for changed in (requests[:1], list(reversed(requests)), requests + requests[:1]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_prepared_requests(changed, expected)
        requests[0].arguments = ("changed", "b")
        with self.assertRaises(ValueError):
            validate_prepared_requests(requests, expected)

    def test_harness_013_readonly_device_property_is_respected(self):
        class Harness013LM:
            def __init__(self):
                self._device = None
                self._rank = 0
                self._world_size = 1

            @property
            def device(self):
                return self._device

        native = adapter()
        with mock.patch.dict(sys.modules, {"lm_eval.api.model": SimpleNamespace(LM=Harness013LM)}):
            wrapped = as_harness_lm(native)
        self.assertIsInstance(wrapped, Harness013LM)
        self.assertEqual(wrapped.device, torch.device("cpu"))
        self.assertEqual(wrapped.loglikelihood([request("a", "b")]), native.loglikelihood([request("a", "b")]))

    def test_harness_wrapper_is_real_lm_when_optional_harness_is_installed(self):
        try:
            from lm_eval.api.model import LM
            from lm_eval import utils
        except ModuleNotFoundError as exc:
            if exc.name != "lm_eval":
                raise
            self.skipTest("lm-eval is intentionally absent from the CPU training environment")
        lm = adapter()
        wrapped = as_harness_lm(lm)
        self.assertIsInstance(wrapped, LM)
        self.assertEqual(wrapped.rank, 0)
        self.assertEqual(wrapped.world_size, 1)
        self.assertEqual(wrapped.device, torch.device("cpu"))
        self.assertEqual(wrapped.loglikelihood([request("a", "b")]), lm.loglikelihood([request("a", "b")]))
        with self.assertRaises(ValueError):
            wrapped.apply_chat_template([])
        for length in range(20):
            tokens = list(range(length))
            expected = [
                utils.make_disjoint_window(window)
                for window in utils.get_rolling_token_windows(
                    token_list=tokens, prefix_token=99, max_seq_len=4, context_len=1,
                )
            ]
            self.assertEqual(list(rolling_windows(tokens, max_length=4, prefix_token_id=99)), expected)


if __name__ == "__main__":
    unittest.main()
