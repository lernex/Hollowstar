# Lernex pre-seed readiness report — source and method notes

Snapshot: 2026-07-14 (America/Denver)

## Reporting job

- Decision: what Lernex should ship, measure, document, and apply to between July 14 and November 19, 2026 to maximize — not guarantee — the chance of raising a pre-seed round.
- Audience: Lernex founder.
- Assumed November scenario: both mobile apps are public; Metis-1.6 is a credible, reproducible technical result; at least 10 external learners have four-week retained behavior; the company and IP records are diligence-ready.
- Probability denominator: one normal cold application or submission from Lernex under that November scenario, unless a row explicitly says the estimate is for a grant or residency. These are analyst estimates, not published acceptance rates. Correlated decisions mean the probabilities must not be added together.

## Current product and model evidence inspected

- `apps/mobile/app.json` in the Lernex repository: iOS bundle identifier is `net.lernex.app`; Android still uses placeholder package id `com.anonymous.lernex` (lines 14-29 at inspection).
- `apps/mobile/package.json`: current mobile stack is Expo 55 / React Native 0.83.2 with iOS and Android run scripts.
- Mobile migration ledger: substantial iOS/Android native work exists, but Android build-project completeness and full native build verification remain open risks.
- Repo search: Lernex has learner analytics and learning-event infrastructure, but no clearly complete acquisition-to-W4-retention funnel was found. The roadmap therefore treats trustworthy funnel instrumentation as an early gate.
- `METIS_1.6_PLAN.md`: design draft; 2.94B total parameters; 0.407B minimum, 0.464B average, and 0.539B maximum active parameters per pass; dynamic expert routing is re-decided on each recursion step; 300B pretraining tokens; estimated 75 days on one RTX PRO 6000 or 38 days on two at the middle MXFP8 scenario, before post-training and release work.

## Official investor and program sources

Observed terms are current as of the snapshot date. “Published amount” means the target publicly states the amount; it does not mean Lernex will receive it.

1. 1517 Fund — exact founder-thesis fit; first checks $50K-$1M, average $400K pre-seed; Medici grants start at $1K: https://www.1517fund.com/ and https://www.1517fund.com/medici
2. Y Combinator — Fall 2026 deadline July 27; $500K standard deal ($125K for 7% plus $375K uncapped MFN SAFE): https://www.ycombinator.com/apply and https://www.ycombinator.com/deal
3. South Park Commons Founder Fellowship — Fall 2026 deadline August 2; $400K for 7% plus $600K guaranteed in the next external round: https://www.southparkcommons.com/founder-fellowship
4. Historical SPC selectivity reference only — inaugural cohort selected 16 fellows / 10 teams from nearly 1,000 founder applications: https://blog.southparkcommons.com/p/kicking-off-the-inaugural-spc-founder-fellowship
5. Afore Grants — rolling; age 21 or under in North America; $1K non-dilutive, 12 weeks: https://grants.afore.vc/
6. Afore Alpha / Founder in Residence — pre-seed checks $500K-$2M+ through Alpha; at least $100K through FIR: https://www.afore.vc/afore-alpha and https://www.afore.vc/residence
7. Z Fellows — rolling, no age restriction, solo accepted; optional $10K investment at a $1B valuation cap: https://www.zfellows.com/
8. Thiel Fellowship — rolling; age 22 or younger, no university degree; $250K over two years, no equity; must leave school to accept: https://thielfellowship.org/faq
9. Founders, Inc. — emerging-tech/AI/consumer; checks up to $250K for 4-7%; its terms expressly allow an under-18 applicant with parent/guardian permission: https://f.inc/about and https://f.inc/terms
10. PearX — about 20 teams; $250K-$2M; solo founders accepted: https://pear.vc/pearx/
11. Antler US — rolling monthly residency; current initial commitment $500K-$1M; solo founders accepted and at least one founder must be full-time in person: https://www.antler.co/location/us
12. Precursor Ventures — no traction requirement; up to $500K; 30-40 new investments per year: https://precursorvc.com/philosophy/
13. Denver Ventures — pre-seed and seed; $250K-$800K: https://denverventures.co/
14. Reach Capital — learning/health/work; initial checks from $100K at pre-seed to $12M later: https://www.reachcapital.com/reach-capital-investment-faqs/
15. Hustle Fund — pre-seed; first check $150K: https://www.hustlefund.vc/faq
16. Techstars — $220K: $20K for 5% plus $200K uncapped MFN SAFE: https://www.techstars.com/newsroom/investment-terms
17. Learn Capital — early and growth stage, education and human-capital focus, including AI-powered learning; no public standard first-check size found: https://www.learn.vc/about
18. Conviction — AI-native, early, often first investor; public check range $1M-$25M: https://www.conviction.com/
19. Air Street Capital — AI-first; public pre-seed/seed range $500K-$5M: https://www.airstreet.com/ai-venture-capital
20. Owl Ventures — education/workforce, every stage; no public standard first-check size found: https://www.owlvc.com/about-us
21. AI Grant — $250K uncapped SAFE, but Batch 4 applications were closed at the snapshot date: https://aigrant.com/

## Store-readiness sources

- Apple organization enrollment requires a legal entity, D-U-N-S number, legal binding authority, a domain email, and a functioning website; the enrolling person must be the age of majority: https://developer.apple.com/help/account/membership/program-enrollment/
- Apple says a new D-U-N-S number may take up to five business days, plus up to two business days to reach Apple: https://developer.apple.com/help/account/membership/D-U-N-S/
- Apps that create accounts must allow account deletion in-app; Sign in with Apple token revocation also needs to be handled: https://developer.apple.com/support/offering-account-deletion-in-your-app
- Google Play organization accounts require a D-U-N-S number and organizational identity documents: https://support.google.com/googleplay/android-developer/answer/13634885
- Google Play policy requires an account-deletion route when users can create accounts: https://support.google.com/googleplay/android-developer/answer/17105854

## Corporate and legal sources

- Every sale of a security — including to one friend or family member — must be registered or exempt; SAFEs and stock are securities: https://www.sec.gov/resources-small-businesses/capital-raising-building-blocks/private-companies-sec and https://www.sec.gov/resources-small-businesses/capital-raising-building-blocks/common-startup-securities
- Earlier noncompliance can cause later investors to walk away: https://www.sec.gov/resources-small-businesses/capital-raising-building-blocks/consequences-noncompliance
- Delaware formation information and fee calculators: https://corp.delaware.gov/howtoform/ and https://corp.delaware.gov/fee/
- IRS Form 15620 / 83(b) elections are due within 30 days after restricted property is transferred: https://www.irs.gov/pub/irs-pdf/f15620.pdf
- IRS Form 8822-B reports a responsible-party change and is due within 60 days: https://www.irs.gov/forms-pubs/about-form-8822-b
- CU Boulder’s Entrepreneurial Law Clinic serves local entrepreneurs/startups and provides free transactional legal help to accepted clients: https://www.colorado.edu/law/academics/clinics/entrepreneurial-law-clinic

## Probability method

- No target except historical SPC publishes a current acceptance rate usable for this exact company. The report therefore uses broad fit-adjusted ranges.
- Inputs: founder/thesis fit; current traction; solo-founder status; age; technical proof; program batch size; open/rolling application path; likely competitive intensity; stage/check mismatch; and whether the investor requires education outcomes, revenue, or institutional-scale evidence.
- “Next step” means interview, residency, partner call, or equivalent serious evaluation.
- “Cash/check” means a grant or investment from that single submission, not the chance of completing an entire syndicated round.
- The estimates assume clear applications, honest numbers, a live demo, no undisclosed cap-table/IP problems, and the November evidence bar in the report. Missing the mobile, retention, or corporate gates should roughly halve the direct-VC estimates.

## Chart map

- Section: retention evidence must start now.
- Question: how much cumulative top-of-funnel work is required before November to show genuine four-week retention?
- Type: two-series line chart, 10 ordered dates.
- Fields: `date`, `series`, `users`, plus retained context fields `target_kind`, `minimum_november_bar`, and `milestone`.
- Takeaway: November cannot manufacture four-week history; the external-learner pipeline must begin in July/August.
- Palette: two approved roots (blue for activated, gold for W4 retained), direct axis/legend labels, no third color.
- Data status: founder operating targets, not observed product data or a forecast.

