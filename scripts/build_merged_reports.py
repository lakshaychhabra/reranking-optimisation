#!/usr/bin/env python3
"""Build the merged four-page brief and concise technical experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from build_final_reports import (
    FIG,
    NAVY,
    PALE_BLUE,
    PALE_GRAY,
    TEAL,
    add_body,
    add_bullets,
    add_figure,
    add_page_break,
    add_page_number,
    add_table,
    borders,
    configure_document,
    set_cell_margins,
    set_core_properties,
    shade,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
GIT_URL = "https://github.com/lakshaychhabra/reranking-optimisation"
HF_URL = "https://huggingface.co/datasets/lakshaychhabra/wg-recommendation-data"
ORANGE = "D97706"
INK = "172238"
MUTED = "5F6B7A"


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def verify_evidence() -> None:
    """Fail generation if a headline value drifts from the frozen artifacts."""
    summary = load_json("results/summary.json")
    models = {row["model"]: row for row in summary["models"]}
    assert models["qwen35_2b_base"]["top1_accuracy"] == 0.22
    assert models["qwen35_2b_sft"]["top1_accuracy"] == 0.84
    assert models["qwen35_2b_sft"]["top3_recall"] == 0.866667
    assert models["qwen35_2b_grpo"]["top1_accuracy"] == 0.84
    assert models["qwen35_2b_grpo"]["top3_recall"] == 0.86
    assert models["qwen"]["top1_accuracy"] == 0.42
    reward = load_json("artifacts/reward_calibration/validation/validation_summary.json")
    assert reward["checks"] == {
        "golden_used_for_calibration": False,
        "llm_calls": 0,
        "permutation_control_passed": True,
        "weights_reselected": False,
    }
    frozen = reward["locked_golden_evaluation"]["frozen"]
    assert frozen["top1_correct"] == 42 and frozen["top3_hits"] == 119
    split = load_json("artifacts/splits/split_manifest.json")
    assert split["checks"]["golden_risk_overlap"] == 0
    assert split["checks"]["disjoint_and_exhaustive"] is True
    teacher = load_json("artifacts/teacher/annotation_summary.json")
    assert teacher["annotation_count"] == 1490
    assert teacher["actual_batch_cost_usd"] == 3.635911
    serve = load_json(
        "artifacts/serving_benchmark/"
        "qwen35-2b-sft-vllm-rtx4090-sustained-20260926/summary.json"
    )
    assert round(serve["best_throughput"]["requests_per_second"], 6) == 63.659166


def style_document(doc: Document, compact: bool) -> None:
    configure_document(doc, compact=compact)
    section = doc.sections[0]
    section.top_margin = Inches(0.48 if compact else 0.58)
    section.bottom_margin = Inches(0.48 if compact else 0.56)
    section.left_margin = Inches(0.58 if compact else 0.68)
    section.right_margin = Inches(0.58 if compact else 0.68)
    normal = doc.styles["Normal"]
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.font.size = Pt(9.15 if compact else 9.7)
    normal.paragraph_format.space_after = Pt(3.5 if compact else 4.2)
    normal.paragraph_format.line_spacing = 1.02 if compact else 1.05
    doc.styles["Title"].font.color.rgb = RGBColor.from_string(NAVY)
    doc.styles["Heading 1"].font.color.rgb = RGBColor.from_string(NAVY)
    doc.styles["Heading 2"].font.color.rgb = RGBColor.from_string(TEAL)
    header = section.header.paragraphs[0]
    header.text = "WG RECOMMENDATION MODEL  /  EXPERIMENT REPORT"
    header.runs[0].font.color.rgb = RGBColor.from_string(MUTED)
    footer = section.footer.paragraphs[0]
    footer.clear()
    footer.add_run("WG Recommendation Model  |  Lakshay Chhabra  |  ")
    add_page_number(footer)


def add_rule(doc: Document, color: str = TEAL, width: str = "18") -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(4)
    p_pr = p._p.get_or_add_pPr()
    borders_node = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), width)
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)
    borders_node.append(bottom)
    p_pr.append(borders_node)


def add_label(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text.upper())
    r.bold = True
    r.font.size = Pt(7.5)
    r.font.color.rgb = RGBColor.from_string(TEAL)


def add_heading(doc: Document, title: str, level: int = 1, kicker: str | None = None) -> None:
    if kicker:
        add_label(doc, kicker)
    doc.add_heading(title, level=level)


def add_callout(doc: Document, title: str, body: str, color: str = PALE_BLUE) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    shade(cell, color)
    set_cell_margins(cell, 105, 125, 105, 125)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(NAVY)
    p = cell.add_paragraph(body)
    p.paragraph_format.space_after = Pt(0)
    borders(table)


def add_kpis(doc: Document, items: list[tuple[str, str]]) -> None:
    table = doc.add_table(rows=1, cols=len(items))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for idx, (value, label) in enumerate(items):
        cell = table.cell(0, idx)
        shade(cell, NAVY if idx % 2 == 0 else TEAL)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell, 90, 60, 80, 60)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(value)
        r.bold = True
        r.font.size = Pt(16)
        r.font.color.rgb = RGBColor(255, 255, 255)
        p = cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(label)
        r.font.size = Pt(7.2)
        r.font.color.rgb = RGBColor(255, 255, 255)
    borders(table)


def add_link(doc: Document, label: str, url: str) -> None:
    p = doc.add_paragraph(style="Small Body")
    p.add_run(f"{label}: ").bold = True
    part = p.part
    rid = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "2C6ECB")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.extend([color, underline])
    run.append(rpr)
    text = OxmlElement("w:t")
    text.text = url
    run.append(text)
    hyperlink.append(run)
    p._p.append(hyperlink)


def page_title(doc: Document, n: str, title: str, subtitle: str) -> None:
    add_label(doc, f"{n} / {subtitle}")
    doc.add_heading(title, level=1)
    add_rule(doc)


def add_stage_table(doc: Document) -> None:
    add_table(
        doc,
        ["Stage", "What we did", "Why", "Verified result"],
        [
            ["Data", "Generated risks + 2,000 engine requests", "Create task-specific panels", "1,490 usable panels / 9,632 tariff offers"],
            ["Teacher", "GPT-6 Sol silver labels", "Scale ranked supervision", "1,490 Top-3 labels"],
            ["Reward", "20k deterministic search", "Align proxy to teacher", "49% → 94% calibration Top-1"],
            ["SFT", "2B BF16 LoRA", "Test smaller-model hypothesis", "22% → 84% golden Top-1"],
            ["GRPO", "10-step continuation", "Test reward optimization", "84% Top-1; no broad gain"],
            ["Serving", "RTX 4090 vLLM benchmark", "Measure deployability", "63.66 rec/s at c64"],
        ],
        widths=[0.72, 1.72, 1.72, 1.87],
        compact=True,
    )


def build_brief(path: Path) -> None:
    doc = Document()
    style_document(doc, compact=True)
    set_core_properties(doc, "WG Recommendation Model Submission Brief", "Four-page executive report")

    add_label(doc, "Executive submission brief / September 2026")
    p = doc.add_paragraph(style="Title")
    p.add_run("Compact model training for\ninsurance tariff ranking")
    p = doc.add_paragraph(style="Report Subtitle")
    p.add_run("Residential building-insurance tariff ranking with SFT, a measured GRPO continuation, and production economics")
    add_rule(doc, ORANGE, "28")
    add_callout(
        doc,
        "Assignment outcome — achieved",
        "The goal was to exceed 80% Top-1 agreement while producing a deployable ranked Top-3. The selected 2B SFT adapter reached 84% Top-1, 86.7% Top-3 recall, and 100% strict JSON on the held-out Fable benchmark.",
    )
    add_kpis(doc, [("0.22 → 0.84", "2B base → SFT Top-1"), ("86.7%", "Top-3 recall"), ("100%", "strict JSON"), ("$7.80", "total billed spend")])
    add_heading(doc, "Task, approach, outcome", kicker="Decision in one page")
    add_stage_table(doc)
    add_figure(doc, "01-dataset-funnel-and-splits.png", "Leakage-safe path from generated risks to training and evaluation", 6.55)
    add_body(doc, "Context: the supplied Qwen3.5-9B reference scored 42% Top-1. The trained-model improvement reported here is correctly measured from our unchanged 2B base (22%) to our 2B SFT model (84%).", style="Source Note")

    add_page_break(doc)
    page_title(doc, "02", "Data and reward: make the objective trustworthy", "Task / What / Why / Result")
    add_table(
        doc,
        ["Task", "What we did", "Why this way", "What we got"],
        [["Build supervision", "2,000 quote-engine requests; GPT-6 Sol labelled the 1,490 usable panels", "Real quote panels preserve product trade-offs; silver labels scale cheaply", "9,632 tariff offers returned; five insurers; nine tariffs; disjoint splits; zero golden-risk overlap"]],
        widths=[1.05, 2.0, 2.0, 1.45], compact=True,
    )
    add_figure(doc, "03-reward-calibration.png", "Reward search on the 200-scenario calibration split", 6.55)
    add_heading(doc, "What the reward audit changed", level=2)
    add_body(doc, "Within panels, coverage count and deductible did not vary; coverage score and annual price carried the consistent ranking signal. A deterministic 20,000-candidate search selected a quality–price curve without LLM calls or golden labels:", style="Small Body")
    add_callout(doc, "Frozen symbolic score", "reward = (0.2863 + 0.4273 × normalized coverage score) / (1 + 1.7936 × normalized price)", PALE_GRAY)
    add_table(
        doc,
        ["Method on golden", "Top-1", "Top-3 hits", "Top-3 recall", "Inference"],
        [["Calibrated symbolic argmax", "42/50 (84%)", "119/150", "79.3%", "None"], ["Selected 2B SFT", "42/50 (84%)", "130/150", "86.7%", "Required"]],
        widths=[1.95, 1.0, 1.05, 1.05, 1.05], compact=True,
    )
    add_callout(doc, "Why not deploy only the zero-cost scorer?", "For a Top-1-only product, it is a compelling simpler baseline. The assignment requires a ranked Top-3: SFT recovered 11 more of the 150 Fable-selected positions and offers a learned policy for interactions beyond the fixed score. The tie is evidence for a strong baseline—not proof that the scorer is ground truth.")

    add_page_break(doc)
    page_title(doc, "03", "Model choice, SFT, and the GRPO stopping decision", "Learned policy")
    add_heading(doc, "Why 2B and why BF16 LoRA?", level=2)
    add_body(doc, "The proposal referenced a 9B model; our hypothesis was that structured tariff ranking should work with substantially less capacity. Qwen3.5-2B tests that claim directly. BF16 LoRA updated 10.9M parameters (0.49%): simple to run, easy to inspect, and sufficient for the available GPU. Keeping the experiment simple was also a reason not to add QLoRA complexity.", style="Small Body")
    add_table(
        doc,
        ["Golden evaluation", "Top-1", "Top-3 recall", "Strict JSON", "Interpretation"],
        [["2B base", "22%", "66.7%", "2%", "Underpowered without task adaptation"], ["2B + SFT", "84%", "86.7%", "100%", "Selected model"], ["2B + SFT + GRPO-10", "84%", "86.0%", "100%", "Stable continuation; no broad gain yet"]],
        widths=[1.35, 0.75, 0.9, 0.9, 2.4], compact=True,
    )
    add_figure(doc, "04-sft-training.png", "SFT produced the decisive quality and format-control improvement", 6.45)
    add_heading(doc, "Why stop GRPO at ten steps?", level=2)
    add_body(doc, "GRPO worked technically: it remained stable and corrected one SFT Top-1 error on the 90-scenario development set (83/90 → 84/90). Ten full Top-3 orderings changed, with a small net recall loss. On golden, SFT and GRPO both reached 42/50 Top-1; GRPO changed one list but not its first choice.", style="Small Body")
    add_callout(doc, "Practical ceiling, not a failed method", "GPT-6 Sol—the silver teacher and reward-calibration target—also achieved 84% Top-1 against Fable, and SFT had already reached that observed level while clearing the assignment’s >80% target. More GRPO steps or samples might still improve the remaining predictions, but optimizing the same silver proxy more aggressively was not justified within the time budget. SFT therefore remains the evidence-backed selection.")

    add_page_break(doc)
    page_title(doc, "04", "Production case, delivery, and conclusion", "Decision")
    add_figure(doc, "07-cost-quality.png", "Held-out quality versus measured or supplied serving cost", 6.45)
    add_table(
        doc,
        ["Serving mode / comparator", "Top-1", "Cost per 1k", "What it means"],
        [["2B SFT, RTX 4090 saturated", "84%", "€0.0028", "Measured 63.66 rec/s at concurrency 64"], ["2B SFT, 10% utilization", "84%", "€0.0283", "Conservative utilization scenario"], ["2B SFT, unbatched", "84%", "€0.0539", "Measured lower-throughput bound"], ["Supplied Qwen3.5-9B", "42%", "€0.1869", "Reference"], ["Gemini / GPT-5.4", "80% / 84%", "€0.8897 / €1.3445", "About 5–7× the 9B reference cost"]],
        widths=[2.0, 0.72, 1.25, 2.5], compact=True,
    )
    add_heading(doc, "Actual experiment spend", level=2)
    add_body(doc, "Teacher selection and batch labelling cost approximately $3.80; the complete RunPod bill for training, evaluation, and benchmarking was $4.00. Total billed cash spend: approximately $7.80.", style="Small Body", bold_lead="Teacher selection")
    add_heading(doc, "Delivered and reproducible", level=2)
    add_link(doc, "Git repository", GIT_URL)
    add_link(doc, "Hugging Face dataset", HF_URL)
    add_body(doc, "The package includes adapters, configs, logs, frozen results, hashes, portable reproduction instructions, and prepared non-golden data. Apple MPS reproduced all 50 selected SFT outputs exactly.", style="Small Body")
    add_callout(doc, "Final decision", "Select the Qwen3.5-2B SFT LoRA. It met the assignment objective—84% Top-1 versus the >80% target—while improving Top-3 recall, enforcing the output contract, and supporting low-cost self-hosted serving. Keep the calibrated scorer as the zero-inference Top-1 fallback and GRPO as a credible next experiment once stronger labels or a new untouched benchmark are available.")
    add_body(doc, "Cost scope: GPU-only serving estimates exclude storage, networking, cold starts, redundancy, and operations. The 50-scenario benchmark is held out but small; results should be confirmed on a larger expert-labelled set before production policy decisions.", style="Source Note")

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


def tech_page(doc: Document, n: str, title: str, subtitle: str, lead: str) -> None:
    add_page_break(doc)
    page_title(doc, n, title, subtitle)
    add_callout(doc, "Bottom line", lead)


def build_technical(path: Path) -> None:
    doc = Document()
    style_document(doc, compact=False)
    set_core_properties(doc, "WG Recommendation Model Technical Experiment Report", "Technical evidence and decisions")

    add_label(doc, "Technical experiment report / September 2026")
    p = doc.add_paragraph(style="Title")
    p.add_run("Compact model training for\ninsurance tariff ranking")
    p = doc.add_paragraph(style="Report Subtitle")
    p.add_run("Reaching top-tier ranking quality with Qwen3.5-2B through supervised fine-tuning, a controlled GRPO continuation, and a verified quality–cost evaluation")
    add_rule(doc, ORANGE, "30")
    add_kpis(doc, [("84%", "held-out Top-1"), ("86.7%", "Top-3 recall"), ("2B", "base parameters"), ("$7.80", "total billed")])
    add_figure(doc, "01-dataset-funnel-and-splits.png", "End-to-end experiment: independent data roles and a locked final benchmark", 6.25)
    add_callout(doc, "Executive conclusion", "The selected SFT LoRA moved the unchanged 2B base from 22% to 84% held-out Top-1, matched the best observed frontier/teacher agreement, produced 100% strict JSON, and served at 63.66 recommendations/s in the sustained benchmark. GRPO was stable and promising on one development error, but SFT had already reached the observed supervision ceiling; ten steps did not establish a broader gain.")
    add_body(doc, "Report map: Task and success criteria → system and data → teacher and reward → model training → evaluation → economics → release and next experiments.", style="Source Note")

    tech_page(doc, "02", "Executive decision dashboard", "What was decided", "Select Qwen3.5-2B SFT LoRA; retain the symbolic scorer as a Top-1 fallback and GRPO as a future experiment, not the production checkpoint.")
    add_stage_table(doc)
    add_heading(doc, "Decision matrix", level=2)
    add_table(doc, ["Option", "Top-1", "Top-3", "Runtime", "Decision"], [["Calibrated scorer", "84%", "79.3%", "None", "Fallback when only rank one matters"], ["2B SFT", "84%", "86.7%", "Low", "Selected"], ["2B GRPO-10", "84%", "86.0%", "Low", "Ablation / future continuation"], ["Frontier APIs", "80–84%", "79.3–88%", "API", "Teacher/comparator, not deployment"]], widths=[1.7,0.75,0.8,0.8,2.4])
    add_heading(doc, "What made the result credible", level=2)
    add_bullets(doc, ["All experimental roles were disjoint; golden-risk overlap was zero.", "The canonical prompt builder controlled teacher, SFT, GRPO, and evaluation inputs.", "Reward weights were frozen before golden evaluation; the golden set was not used for tuning.", "Claims in this report are tied to machine-readable summaries, run records, and hashes."])
    add_link(doc, "Git repository", GIT_URL)
    add_link(doc, "Hugging Face dataset", HF_URL)

    tech_page(doc, "03", "Task and success criteria", "Problem contract", "Given a building risk and its priced quote panel, return three distinct offered tariffs in ranked JSON; the primary target was held-out Top-1 agreement above 80%.")
    add_table(doc, ["Metric", "Definition", "Why it matters"], [["Top-1 agreement", "First recommendation equals reference rank one", "Primary assignment measure"], ["Top-3 recall", "Reference tariffs recovered across three slots", "Measures shortlist usefulness"], ["Strict JSON", "Exact required schema", "Operational reliability"], ["Reward ratio", "Supplied reward of model rank one / scenario maximum", "Audits quality–price trade-off; not ground truth"], ["Serving cost", "Measured GPU cost per 1,000", "Deployment economics"]], widths=[1.35,2.7,2.45])
    add_heading(doc, "Reference hierarchy", level=2)
    add_body(doc, "Fable rankings are the held-out evaluation reference. GPT-6 Sol is scalable silver supervision. The calibrated numeric reward is a deterministic approximation to that teacher. The objects are related, but none should be presented as objective truth.")
    add_callout(doc, "Important correction", "The supplied 9B model scored 42%. Our experiment did not fine-tune that model: our unchanged 2B base scored 22%, and our 2B SFT checkpoint scored 84%. Therefore the training gain is 0.22 → 0.84; 0.42 is a separate assignment reference.", PALE_GRAY)
    add_heading(doc, "Output contract", level=2)
    add_body(doc, '{"top_3":[{"rank":1,"tariff_id":"49"},{"rank":2,"tariff_id":"48"},{"rank":3,"tariff_id":"162"}]}', style="Small Body")

    tech_page(doc, "04", "System and experimental pipeline", "How the work fits together", "One canonical prompt and explicit data roles connected scenario generation, real quote collection, teacher labels, reward calibration, SFT, GRPO, and evaluation.")
    add_figure(doc, "01-dataset-funnel-and-splits.png", "Dataset funnel and experimental separation", 6.45)
    add_table(doc, ["Control", "Implementation", "Risk controlled"], [["Scenario identity", "Hash of normalized risk JSON", "Duplicate leakage"], ["Quote access", "Cached client; ≥3 s live-call spacing", "Cost and rate safety"], ["Prompt construction", "src/prompt.py only", "Training/evaluation drift"], ["Quote order", "Seeded shuffle with stored mapping", "Positional bias"], ["Golden boundary", "Final evaluation only", "Selection leakage"]], widths=[1.45,2.5,2.55])
    add_heading(doc, "Why this architecture", level=2)
    add_body(doc, "The design separates four questions: Can we create representative quote panels? Can a teacher provide consistent rankings? Can a deterministic reward capture the teacher’s preference? Can a small learned policy generalize beyond that formula? This makes failures attributable rather than mixing data, objective, and optimization effects.")

    tech_page(doc, "05", "Dataset creation and coverage", "Inputs", "The final corpus contains 1,490 usable scenarios and 9,632 live-engine quotes, then assigns every scenario exactly one experimental role.")
    add_figure(doc, "02-dataset-distribution.png", "Scenario and quote-panel coverage in the generated corpus", 6.35)
    add_table(doc, ["Split", "N", "Labels", "Purpose"], [["Reward calibration", "200", "Teacher", "Reward selection and validation"], ["SFT train / validation", "300 / 90", "Teacher", "Imitation and checkpoint selection"], ["GRPO train / validation", "810 / 90", "Train labels omitted", "Policy optimization and offline gate"], ["Golden", "50", "Fable", "Final reporting only"]], widths=[1.7,1.0,1.7,2.8])
    add_heading(doc, "Checks", level=2)
    add_bullets(doc, ["1,490 assignments were disjoint and exhaustive.", "No generated risk exactly overlapped a golden risk.", "Final panels span five insurers and nine tariff IDs.", "Only panels with at least three offered quotes entered training."])
    add_body(doc, "Scope note: generated risks and the finite product catalogue support a controlled experiment, not a claim of complete production-population coverage.", style="Source Note")

    tech_page(doc, "06", "Teacher model and supervision", "Silver labels", "GPT-6 Sol produced 1,490 ranked Top-3 labels for $3.64 in the batch run; including teacher selection, the label programme cost approximately $3.80.")
    add_table(doc, ["Item", "Verified value"], [["Teacher", "GPT-6 Sol"], ["Annotations", "1,490"], ["Input / output tokens", "1,546,926 / 417,797"], ["Batch annotation cost", "$3.635911"], ["Teacher selection + labels", "≈$3.80"], ["Observed golden Top-1", "84% (42/50)"]], widths=[2.3,4.1])
    add_heading(doc, "Why use a teacher", level=2)
    add_body(doc, "Direct Fable labels were available only for the locked 50-scenario benchmark. A frontier teacher made task-specific supervision affordable at 1,490-scenario scale. The same canonical prompt and quote-ID mapping were used throughout, and the raw numeric reward was not exposed to the teacher.")
    add_heading(doc, "What the 84% teacher result means", level=2)
    add_body(doc, "It is an observed ceiling for this supervision pipeline, not a theorem about all possible models. SFT matching 84% indicates that additional optimization against the same teacher/reward proxy had limited demonstrated headroom against Fable. Better expert labels could change that ceiling.")
    add_callout(doc, "Economic motivation", "A frontier/Fable-quality API route, represented by the evaluated Gemini and GPT-5.4 comparators, was materially more expensive than the supplied 9B reference: about 5× and 7× its per-recommendation cost. This is a comparator-based estimate, not a measured Fable tariff. Teacher-supervised fine-tuning aimed to retain that ranking quality in a self-hostable 2B model.", PALE_GRAY)

    tech_page(doc, "07", "Reward grid search and validation", "Objective design", "Reward calibration showed that the effective task was a quality–price trade-off; a frozen deterministic scorer matched 84% Fable Top-1 but remained weaker on the full Top-3.")
    add_figure(doc, "03-reward-calibration.png", "Twenty-thousand-candidate search and held-out controls", 6.4)
    add_table(doc, ["Evaluation", "Default", "Frozen calibrated"], [["Calibration Top-1", "49%", "94%"], ["Calibration Top-3 recall", "68.8%", "84.2%"], ["Golden Top-1", "38%", "84%"], ["Golden Top-3 recall", "65.3%", "79.3%"], ["Golden pairwise", "66.0%", "90.4%"]], widths=[2.8,1.75,1.85])
    add_callout(doc, "Frozen formula", "reward = (0.2863 + 0.4273 × scoreₙ) / (1 + 1.7936 × priceₙ)", PALE_GRAY)
    add_body(doc, "The selected weights were unchanged during validation; calibration used no golden labels and no LLM calls. A permutation negative control returned chance-level pairwise ranking, reducing the risk that the search merely manufactured agreement.", style="Small Body")

    tech_page(doc, "08", "Model selection and training design", "Why this model", "Qwen3.5-2B was chosen to challenge the assumption that a 9B model was necessary; BF16 LoRA kept the experiment transparent and operationally simple.")
    add_table(doc, ["Choice", "Selected", "Reason"], [["Base model", "Qwen3.5-2B", "Small enough to test the capacity hypothesis; modern instruction base"], ["Adaptation", "BF16 LoRA", "Simple pipeline; no quantization confound; fits available GPU"], ["LoRA", "rank 16 / alpha 16 / dropout 0", "10.9M trainable parameters (0.49%)"], ["SFT data", "300 train / 90 validation", "Independent teacher-labelled pool"], ["SFT schedule", "3 epochs / 21 steps", "Validation-loss checkpoint selection"], ["Loss", "Completion only", "Learn output policy without fitting prompt tokens"]], widths=[1.45,2.1,3.0])
    add_heading(doc, "Why not QLoRA", level=2)
    add_body(doc, "The full BF16 base plus LoRA fit the selected hardware, so quantization was not required. Avoiding QLoRA kept the experiment simpler and removed another variable from a short, decision-oriented comparison.")
    add_heading(doc, "What would change the model choice", level=2)
    add_body(doc, "If time allowed, the next capacity experiment would test ≤1B models under the same split and evaluation contract. The 2B result supports—not exhausts—the hypothesis that this structured task needs far less than 9B parameters.")

    tech_page(doc, "09", "Supervised fine-tuning", "Main model gain", "SFT transformed both ranking quality and operational reliability: 22% → 84% Top-1 and 2% → 100% strict JSON.")
    add_figure(doc, "04-sft-training.png", "SFT learning curve and selected checkpoint", 6.4)
    add_table(doc, ["Golden metric", "2B base", "2B SFT", "Change"], [["Top-1", "22%", "84%", "+62 pp"], ["Top-3 recall", "66.7%", "86.7%", "+20 pp"], ["Strict JSON", "2%", "100%", "+98 pp"], ["Valid three unique IDs", "100%", "100%", "Maintained"]], widths=[2.2,1.25,1.25,1.7])
    add_heading(doc, "Interpretation", level=2)
    add_body(doc, "The improvement is not merely format compliance: Top-1 and Top-3 both rose sharply. The selected model’s lower supplied-reward ratio than price-dominant baselines reflects a deliberate shift toward the teacher/Fable quality preference, not an internal contradiction.")
    add_callout(doc, "Selection rule", "The SFT checkpoint was selected from validation loss on its own 90-scenario split. Golden results were reported afterward and were not used to choose SFT steps or hyperparameters.", PALE_GRAY)

    tech_page(doc, "10", "GRPO continuation: useful signal, limited evidence", "Policy optimization", "GRPO did not fail: it was stable and fixed one development Top-1 error. Ten steps were insufficient to demonstrate a broad improvement beyond an SFT model already at the teacher’s observed Fable agreement.")
    add_figure(doc, "05-grpo-training.png", "Ten-step GRPO training diagnostics", 6.35)
    add_table(doc, ["Evaluation", "SFT", "GRPO-10", "Observed change"], [["Development Top-1", "83/90", "84/90", "+1 corrected scenario"], ["Development Top-3 recall", "87.0%", "86.3%", "Small net loss"], ["Golden Top-1", "42/50", "42/50", "No change"], ["Golden Top-3 recall", "86.7%", "86.0%", "One fewer hit / 150"], ["Strict JSON", "100%", "100%", "No degradation"]], widths=[2.0,1.15,1.15,2.1])
    add_heading(doc, "Why stop", level=2)
    add_bullets(doc, ["The assignment’s >80% objective was already exceeded by SFT.", "SFT matched the teacher’s observed 84% Fable Top-1.", "Additional optimization would still target a silver reward proxy, not independent truth.", "The time budget favored verifying reproducibility and serving economics over an open-ended RL sweep."])
    add_body(doc, "Longer training, more samples per prompt, or better supervision could still correct remaining cases. The defensible claim is that ten steps did not establish a broad gain—not that GRPO cannot work.", style="Source Note")

    tech_page(doc, "11", "Final evaluation and model selection", "What we got", "The selected 2B SFT model matched the best observed Top-1 result and produced the strongest balanced Top-3 among the compact deployable options.")
    add_table(doc, ["System", "Top-1", "Top-3 recall", "Reward ratio", "Cost / 1k"], [["Supplied Qwen3.5-9B", "42%", "66.0%", "0.983", "€0.1869"], ["Our Qwen3.5-2B base", "22%", "66.7%", "0.844", "—"], ["Calibrated scorer", "84%", "79.3%", "0.812", "€0 inference"], ["Our 2B SFT — selected", "84%", "86.7%", "0.812", "€0.0028–0.0539"], ["Our 2B GRPO-10", "84%", "86.0%", "0.812", "Same serving class"], ["Gemini", "80%", "88.0%", "0.763", "€0.8897"], ["GPT-5.4", "84%", "79.3%", "0.802", "€1.3445"], ["GPT-6 Sol", "84%", "86.7%", "0.832", "€2.8707"]], widths=[2.15,0.85,1.05,1.05,1.35])
    add_heading(doc, "Selection rationale", level=2)
    add_body(doc, "SFT ties the strongest Top-1, matches GPT-6 Sol Top-3, exceeds the symbolic scorer by 11 recovered Top-3 positions, satisfies the output contract, and has a measured low-cost serving path. GRPO adds complexity without a demonstrated aggregate benefit at the tested budget.")
    add_body(doc, "The calibrated scorer remains valuable: if a product consumes only rank one, it offers the same measured Top-1 with no inference. It is not chosen for this deliverable because the required output is a useful ranked Top-3.", style="Small Body")

    tech_page(doc, "12", "Serving benchmark and economics", "Production case", "A dedicated post-fix vLLM benchmark—not incidental training-run timing—provides the serving figures used for the cost claim.")
    add_figure(doc, "09-serving-efficiency.png", "Measured throughput, latency, and utilization on an RTX 4090", 6.35)
    add_table(doc, ["Scenario", "Cost / 1k", "Status"], [["Saturated, concurrency 64", "€0.002832", "Measured; 63.66 rec/s"], ["50% utilization", "€0.005663", "Scenario"], ["25% utilization", "€0.011327", "Scenario—not a recommendation"], ["10% utilization", "€0.028317", "Scenario"], ["Unbatched / concurrency 1", "€0.053899", "Measured lower-throughput mode"]], widths=[2.65,1.4,2.6])
    add_heading(doc, "Timing caveat", level=2)
    add_body(doc, "Early end-to-end runs took longer because FlashAttention was not functioning correctly; that environment issue was later fixed. For this reason, the report does not infer production latency from training or ad-hoc evaluation timestamps. It uses the separate serving benchmark: 1,200 requests, p50 1.003 s, p95 1.105 s, 90.7% mean GPU utilization, and 100% valid outputs.")
    add_heading(doc, "Cost boundaries", level=2)
    add_body(doc, "Serving estimates use a $0.74/hour RTX 4090 and are GPU-only. Storage, networking, cold starts, redundancy, monitoring, and engineering operations are excluded.", style="Source Note")

    tech_page(doc, "13", "Spend, reproducibility, and delivery", "Can it be rerun", "The complete billed experiment cost approximately $7.80, and the release package preserves adapters, configs, logs, frozen outputs, hashes, and a portable reproduction flow.")
    add_table(doc, ["Spend item", "Amount", "Evidence"], [["Teacher pilot / selection", "≈$0.16", "Selection records"], ["Teacher batch labels", "$3.635911", "annotation_summary.json"], ["RunPod total", "$4.00", "Owner-confirmed total bill"], ["Total", "≈$7.80", "Experiment + benchmarking"]], widths=[2.5,1.2,2.9])
    add_heading(doc, "Release contents", level=2)
    add_bullets(doc, ["SFT and GRPO adapters with hashes and resolved configs.", "Frozen base, SFT, GRPO, comparator, and serving summaries.", "Prepared non-golden data and split manifests.", "HOW_TO_RUN.md and portable reproduction script.", "Apple MPS verification: all 50 SFT raw outputs matched CUDA exactly."])
    add_link(doc, "Git repository", GIT_URL)
    add_link(doc, "Hugging Face dataset", HF_URL)
    add_callout(doc, "Reproduction rule", "Use development data for routine checks. The golden benchmark remains final-report evidence and must not become an iterative checkpoint-selection loop.", PALE_GRAY)

    tech_page(doc, "14", "Limits and next experiments", "What remains", "The current evidence supports the assignment decision; production confidence requires stronger independent labels, more scenarios, and a capacity study below 2B.")
    add_table(doc, ["Limitation", "Why it matters", "Next experiment"], [["50-scenario golden set", "One case moves Top-1 by 2 pp", "Larger untouched expert-labelled benchmark"], ["Silver supervision", "Teacher can reproduce its own preferences", "Expert adjudication on disagreements"], ["Synthetic risks / finite catalogue", "May miss operational tail cases", "Shadow evaluation on production-like traffic"], ["Fixed reward formula", "Can miss feature interactions", "Compare learned reward or preference model"], ["2B is smallest tested", "Capacity floor remains unknown", "Repeat SFT at ≤1B"], ["GRPO only 10 steps", "Longer run may improve remaining errors", "Predeclare dev gate; increase steps/samples after better labels"]], widths=[1.75,2.35,2.6])
    add_heading(doc, "Recommended sequence", level=2)
    add_bullets(doc, ["Deploy or shadow-test the selected SFT adapter with the symbolic scorer as a monitoring baseline.", "Collect expert labels specifically where SFT, scorer, and teacher disagree.", "Run the ≤1B capacity ablation under the identical prompt and split contract.", "Only then extend GRPO with a frozen development rule and a new untouched final benchmark."])
    add_callout(doc, "Final conclusion", "The assignment aim was achieved: a small self-hostable model exceeded 80% Top-1 and matched the observed top-tier result at a fraction of serving cost. The main lesson is not that one optimizer won; it is that disciplined data separation, reward interpretation, and simple SFT captured nearly all demonstrated value, while GRPO remains a credible path once the supervision ceiling is raised.")

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--brief", type=Path, default=REPORTS / "WG_Recommendation_Submission_Brief.docx")
    parser.add_argument("--technical", type=Path, default=REPORTS / "WG_Recommendation_Technical_Report.docx")
    args = parser.parse_args()
    _ = args.seed
    verify_evidence()
    build_brief(args.brief)
    build_technical(args.technical)
    print(args.brief)
    print(args.technical)


if __name__ == "__main__":
    main()
