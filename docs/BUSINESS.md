# Plexus Finder: product and business notes

Status: research prototype. Written 18 October 2026. Not legal or regulatory advice.

## What the product actually is

The model finds the brachial plexus on an interscalene ultrasound frame. Clinicians use that view
to place a regional anaesthesia block. So the product is **nerve-block guidance software**, not a
diagnostic tool. It diagnoses nothing. Calling it diagnostic would change the regulatory category
and invite claims the evidence cannot support.

## The market already exists

| Product | Company | Status | Notes |
|---|---|---|---|
| ScanNav Anatomy PNB | Intelligent Ultrasound, now GE HealthCare | FDA De Novo, Oct 2022 | 9 to 10 block regions. GE bought the clinical AI business for about $53M. |
| Nerveblox | SmartAlpha | FDA 510(k) K250818, Aug 2025; CE, UKCA, TGA | 50 structures, runs on GE Venue systems. |

The ScanNav De Novo created FDA product code **QRG**, "Ultrasound Guided Nerve Block Assist", Class
II with special controls. A newcomer can now use the 510(k) route with a predicate, which is cheaper
and faster than De Novo. Nerveblox took 150 days.

Sources: [Fierce Biotech on ScanNav](https://www.fiercebiotech.com/medtech/fda-clears-intelligent-ultrasounds-ai-real-time-guidance-nerve-block-injections),
[Radiology Business on the GE deal](https://radiologybusiness.com/topics/artificial-intelligence/ge-healthcare-completes-53m-acquisition-ai-ultrasound-business),
[AuntMinnie on Nerveblox](https://www.auntminnie.com/clinical-news/ultrasound/news/15753479/us-fda-clears-smartalphas-ai-for-guiding-nerve-blocks),
[510(k) K250818](https://510k.innolitics.com/device/K250818).

## Where a small business could fit

Competing head-on with GE-owned software in high-income hospitals is a poor first move. Three
openings look more realistic, roughly in order of how fast they could earn money.

1. **Training and simulation, not clinical use.** Residency programmes and anaesthesia courses
   need sonoanatomy teaching. A tool that scores a trainee's own scans against the model and an
   expert is educational software and is generally not a medical device. This can sell now.
2. **Annotation and research tooling.** Groups building their own datasets need fast first-pass
   masks plus label-quality checks. The near-duplicate contradiction audit in this repo is a
   real, sellable feature.
3. **Low-resource clinical settings, later.** Point-of-care ultrasound is spreading where regional
   anaesthesia expertise is thin. That is the strongest impact case, and it fits a Wits/South Africa
   base, but it needs the full regulated path (SAHPRA here, plus CE or FDA) and local clinical data.

## What stands between this prototype and any clinical product

1. **Data rights.** The training data comes from the 2016 Kaggle competition. Its licence terms
   were not confirmed for this note. Assume no commercial use until the competition rules are read
   and, if needed, the sponsor gives permission. A commercial model needs its own consented data.
2. **New data.** Multi-site, multi-scanner frames from consenting patients under ethics approval,
   with at least two independent expert annotators per frame so inter-rater agreement is known.
3. **Prospective validation.** The current results are retrospective on one public dataset. A
   cleared product needs a pre-specified study showing that clinicians identify anatomy better or
   faster with the tool.
4. **Quality system.** ISO 13485 quality management, IEC 62304 software lifecycle, ISO 14971 risk
   management and a predetermined change control plan for model updates.
5. **Live video.** Clinicians scan continuously. The product must run on a video feed with stable
   frame-to-frame output, which this prototype does not do yet.

## Suggested next steps

1. Read the Kaggle competition rules and record the data licence position in this file.
2. Show the prototype to two or three regional anaesthetists and one anaesthesia educator. Ask what
   they would pay for in teaching, not in theatre.
3. If the teaching use holds up, scope a small ethics application for a consented, multi-annotator
   dataset at a Johannesburg teaching hospital.
