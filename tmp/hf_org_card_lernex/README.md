---
title: Lernex
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
---

<img
  src="https://huggingface.co/spaces/Lernex/README/resolve/main/assets/lernex-org-banner.jpg"
  alt="Lernex turns learning material into adaptive study paths and compact AI systems"
  width="100%"
/>

_Welcome to the official Lernex organization on Hugging Face._

[Lernex](https://www.lernex.net/) is an AI-personalized learning platform built around a simple idea: studying should adapt to the learner, the material, and the moment. Lernex turns notes, PDFs, slides, classroom material, questions, and learner goals into lessons, quizzes, flashcards, explanations, study paths, feedback, and memory-aware tutoring.

This organization is where Lernex publishes the Metis model line and related open research artifacts. Metis models are compact, efficient language models built for the specific jobs inside a learning product: tutoring, explanations, writing support, practice generation, concise reasoning, and fast product loops that cannot depend on oversized systems for every interaction.

## What Lernex Is Building

- **A continuous learning companion**: Lernex is designed so a learner does not restart from zero every time they open a different class, generated lesson, quiz, flashcard set, or study path.
- **Adaptive study material**: Source material can become structured lessons, checks for understanding, retrieval-friendly notes, and targeted practice.
- **Personalized learning paths**: Lernex uses learner goals, progress, confusion signals, and prior activity to shape what should come next.
- **Fast learning workflows**: Learning tools should feel responsive enough for everyday studying, not like heavyweight one-off AI demos.

## Why We Build Models

Education products need a different model shape than general chat products. Some proprietary models are powerful but expensive or slow for high-frequency learning loops. Some open models are affordable but miss the tutoring, instruction-following, reasoning, or product-specific behavior Lernex needs.

Metis is our answer to that mismatch: build compact models that are useful where learning quality matters, efficient enough to run often, and honest about the tradeoffs between size, reasoning, latency, and cost.

## Featured Models

- **[Metis-1.4 Base](https://huggingface.co/Lernex/Metis-1.4-base)** - current base release for the Metis-1.4 line.
- **[Metis-1.4 Chat](https://huggingface.co/Lernex/Metis-1.4-chat)** - conversational variant for tutoring, writing, explanations, and learning support.
- **[Metis-1.4 Think](https://huggingface.co/Lernex/Metis-1.4-think)** - reasoning-oriented variant for compact step-by-step problem solving.
- **[Metis-1.3 Base](https://huggingface.co/Lernex/Metis-1.3-base)**, **[Chat](https://huggingface.co/Lernex/Metis-1.3-chat)**, and **[Think](https://huggingface.co/Lernex/Metis-1.3-think)** - earlier hybrid Mamba-attention research releases.

## Research Direction

- **Compact tutoring models**: smaller models that can still explain clearly, generate useful practice, and behave consistently in learning contexts.
- **Efficient reasoning**: model variants that spend compute where step-by-step problem solving matters instead of making every request expensive.
- **Product-shaped evaluation**: evaluating models on whether they help learners understand, practice, revise, and move forward, not only on broad benchmark prestige.
- **Deployment practicality**: keeping model size, serving cost, latency, and memory footprint in view from the start.

## How This Connects to Lernex

Lernex uses AI to help learners bring messy material into a system that can teach from it. The long-term goal is not just a better chatbot; it is one learning relationship that carries context across generated lessons, study paths, practice, feedback, and review.

The Metis line supports that goal by giving Lernex a model family it can tune around real learning workflows instead of treating every AI job as the same generic chat problem.

## Resources

- Website: [lernex.net](https://www.lernex.net/)
- Company page: [lernex.net/company](https://www.lernex.net/company)
- Product feedback: [feedback@lernex.net](mailto:feedback@lernex.net)
- Business contact: [business@lernex.net](mailto:business@lernex.net)
