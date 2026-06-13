"""Expert clinical encoder for diabetes inpatient encounters (Tier-0).

Designed as a clinical-informatics specialist would: it does not emit flat
attribute=value triples. It encodes the *clinical relationships* an
endocrinologist reasons over — diagnosis comorbidity over the ICD-9 hierarchy,
the specific diabetes complication, antidiabetic therapy by drug CLASS with
titration, glycemic control state, care acuity/chronicity — and the higher-order
causal/temporal chains that link them (poor control -> therapy escalation;
diabetes -> end-organ complication; chronic utilization -> acute admission).

All mappings are real medical knowledge (ICD-9 category grouping per Strack et
al. 2014; antidiabetic drug classes per pharmacology; ADA glycemic thresholds).
NOTHING is fitted to the readmission label, which is never encoded.
"""
from __future__ import annotations

from sma.ir.schema import Case, Statement, entity, make_case, stmt

from .base import EncodeResult

# --- ICD-9 diagnosis grouping (Strack et al. 2014, the canonical Diabetes-130
#     categorization). Maps a code to a body-system category. ---------------
_DIABETES_COMPLICATION = {
    "0": "uncomplicated", "1": "ketoacidosis", "2": "hyperosmolar",
    "3": "other_coma", "4": "renal", "5": "ophthalmic", "6": "neurological",
    "7": "peripheral_circulatory", "8": "other_specified", "9": "unspecified",
}


def icd9_category(code: str) -> str | None:
    code = (code or "").strip()
    if not code:
        return None
    if code[0] in "EV":                 # external-cause / supplementary -> Other
        return "other"
    if code.startswith("250"):
        return "diabetes"
    try:
        num = int(float(code))
    except ValueError:
        return "other"
    if 390 <= num <= 459 or num == 785:
        return "circulatory"
    if 460 <= num <= 519 or num == 786:
        return "respiratory"
    if 520 <= num <= 579 or num == 787:
        return "digestive"
    if 800 <= num <= 999:
        return "injury"
    if 710 <= num <= 739:
        return "musculoskeletal"
    if 580 <= num <= 629 or num == 788:
        return "genitourinary"
    if 140 <= num <= 239:
        return "neoplasm"
    return "other"


def diabetes_complication(code: str) -> str | None:
    code = (code or "").strip()
    if not code.startswith("250"):
        return None
    _, _, dec = code.partition(".")
    return _DIABETES_COMPLICATION.get(dec[:1], "uncomplicated") if dec else "uncomplicated"


# --- Antidiabetic pharmacology: drug -> therapeutic class -------------------
_DRUG_CLASS = {
    "metformin": "biguanide",
    "glimepiride": "sulfonylurea", "glipizide": "sulfonylurea",
    "glyburide": "sulfonylurea", "chlorpropamide": "sulfonylurea",
    "tolbutamide": "sulfonylurea", "acetohexamide": "sulfonylurea",
    "tolazamide": "sulfonylurea",
    "repaglinide": "meglitinide", "nateglinide": "meglitinide",
    "pioglitazone": "tzd", "rosiglitazone": "tzd", "troglitazone": "tzd",
    "acarbose": "agi", "miglitol": "agi",
    "insulin": "insulin",
    "examide": "other", "citoglipton": "other",
    "glyburide-metformin": "combination", "glipizide-metformin": "combination",
    "glimepiride-pioglitazone": "combination",
    "metformin-rosiglitazone": "combination", "metformin-pioglitazone": "combination",
}
_DRUGS = tuple(_DRUG_CLASS)


def _glycemic_state(a1c: str, glu: str) -> str:
    a1c, glu = (a1c or "").strip(), (glu or "").strip()
    if a1c in (">7", ">8") or glu in (">200", ">300"):
        return "uncontrolled"
    if a1c == "Norm" or glu == "Norm":
        return "controlled"
    return "unmeasured"


def _bin_count(n: int, lo: int = 1, hi: int = 3) -> str:
    if n <= 0:
        return "none"
    return "low" if n <= hi else "high"


def _int(fields: dict, key: str) -> int:
    try:
        return int(float(fields.get(key, "0")))
    except (ValueError, TypeError):
        return 0


class HealthcareEncoder:
    adapter_id = "healthcare"
    version = "1.0.0"

    def encode(self, artifact: str, **kwargs) -> EncodeResult:
        import json
        fields = json.loads(artifact) if isinstance(artifact, str) else dict(artifact)
        return EncodeResult(self.encode_record(fields), ())

    def encode_record(self, f: dict) -> Case:
        # Clinical features are lifted into FUNCTORS (a structure-mapping memory
        # discriminates on functor identity, not entity arguments), each over a
        # constant patient node; the higher-order clinical relations connect
        # them. So two patients with the same comorbidity/therapy/control profile
        # share functors (MAC discriminates) AND share relational structure (SME
        # systematicity).
        p = entity("patient", "patient")
        S: list[Statement] = []

        def feat(name: str) -> Statement:
            return stmt(name, p)

        # 0. Raw discriminative features (high-cardinality functors = retrieval
        #    discrimination). A structure-mapping memory needs BOTH discriminative
        #    detail AND curated structure; abstraction alone collapses patients.
        for col, val in f.items():
            S.append(stmt(col, p, entity(str(val), "value")))

        # 1. Diagnoses -> dx<Category> functors; diabetes complication -> cx<Type>.
        cats: list[str] = []
        dia_stmt: Statement | None = None
        for col in ("diag_1", "diag_2", "diag_3"):
            cat = icd9_category(f.get(col, ""))
            if not cat:
                continue
            d = feat("dx" + cat.capitalize())
            S.append(d)
            cats.append(cat)
            if cat == "diabetes" and dia_stmt is None:
                dia_stmt = d
                comp = diabetes_complication(f.get(col, ""))
                if comp and comp not in ("uncomplicated", "unspecified"):
                    c = feat("cx" + comp.replace("_", " ").title().replace(" ", ""))
                    S.append(c)
                    S.append(stmt("manifests", d, c))                  # higher-order
        # comorbidity: relate the first diagnosis to each distinct other system
        seen_dx = {s.functor: s for s in S if s.functor.startswith("dx")}
        dx_list = list(seen_dx.values())
        for other in dx_list[1:]:
            S.append(stmt("comorbidWith", dx_list[0], other))          # higher-order

        # 2. Therapy -> rx<Class> functors, titration -> up/down<Class>; treats.
        therapy_stmts: dict[str, Statement] = {}
        for drug in _DRUGS:
            status = (f.get(drug, "No") or "No").strip()
            if status == "No":
                continue
            cls = _DRUG_CLASS[drug]
            if cls not in therapy_stmts:
                t = feat("rx" + cls.capitalize())
                therapy_stmts[cls] = t
                S.append(t)
                if dia_stmt is not None:
                    S.append(stmt("treats", t, dia_stmt))              # higher-order
            if status == "Up":
                S.append(feat("titrUp" + cls.capitalize()))
            elif status == "Down":
                S.append(feat("titrDown" + cls.capitalize()))
        if len(therapy_stmts) >= 2:
            S.append(feat("polytherapy"))

        # 3. Glycemic control -> gly<State> functor; clinical picture links.
        gly = _glycemic_state(f.get("A1Cresult", ""), f.get("max_glu_serum", ""))
        gly_stmt = feat("gly" + gly.capitalize())
        S.append(gly_stmt)
        if dx_list:
            S.append(stmt("presentsWith", dx_list[0], gly_stmt))       # higher-order
        for t in therapy_stmts.values():
            S.append(stmt("addresses", t, gly_stmt))                   # higher-order

        # 4. Poor control -> therapy escalation (the causal chain).
        change = (f.get("change", "No") or "No").strip()
        insulin = (f.get("insulin", "No") or "No").strip()
        if gly == "uncontrolled" and (change == "Ch" or insulin in ("Up", "Steady")):
            esc = feat("escalation")
            S.append(esc)
            S.append(stmt("cause", gly_stmt, esc))                     # higher-order

        # 5. Care acuity / chronicity -> functors.
        prior = _int(f, "number_inpatient") + _int(f, "number_emergency") + _int(f, "number_outpatient")
        util = feat("priorUtil" + _bin_count(prior).capitalize())
        S.append(util)
        adm = feat("admit" + str(f.get("admission_type_id", "0")))
        S.append(adm)
        if _bin_count(prior) == "high":
            S.append(stmt("chronicity", util, adm))                    # higher-order
        S.append(feat("los" + _bin_count(_int(f, "time_in_hospital"), hi=4).capitalize()))

        return make_case(S or [feat("encounterUnknown")],
                         {"adapter": self.adapter_id, "tier": 0})
