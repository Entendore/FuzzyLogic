#!/usr/bin/env python3
"""
fuzzy8.py — v8 feature pack for the fuzzydb stack.

Requires: fuzzydb.py (v5) + fuzzy7.py on the path; fuzzynl.py (v6)
enables the dialogue features; pandas is optional.

    from fuzzy8 import Database8 as FuzzyDB
    db = FuzzyDB("company.db")

    # 1) clarification dialogue (answers persist into the lexicon)
    r = db.ask("employees", "very seasond folks")            # console prompt
    r = db.ask("employees", "...", answer_fn=my_fn)          # scripted/API

    # 2) self-explanation
    db.why_not("employees", "age IS young AND salary IS high", {"name": "Grace"})
    db.which_queries_include("employees", {"name": "Elena"})

    # 3) per-user preference profiles
    db.learn_for("recruiter", "employees", where, liked, disliked)
    db.personal_query("recruiter", "employees", where)

    # 4) AHP weights from pairwise comparisons
    w = db.ahp_weights([("salary", "age", "strongly"), ...])

    # 5) operational alerts
    db.define_alert("hc", "count", table="employees",
                    where="salary IS high", above=2)
    db.run_checks()                       # or db.watch(interval=60)

    # 6) time travel
    db.snapshot("employees", "before_zoe")
    db.as_of_query("employees", "salary IS high", as_of="before_zoe")
    db.diff_as_of("employees", "before_zoe")

    # 7) pandas
    df = db.to_dataframe(db.fuzzy_query("employees", "salary IS high"))
    db.from_dataframe(df, "imported")

    # 8) web UI (heatmap + why-button)
    python fuzzy8.py serve company.db 8200

CLI:  python fuzzy8.py            demo
      python fuzzy8.py selftest   built-in assertions
      python fuzzy8.py serve f.db 8200   web console
"""

import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import datetime as _dt

from fuzzydb import (FuzzyError, FuzzySyntaxError, _parse_cached,
                     collect_leaves, eval_tree, TNORMS, SNORMS, LEAF_KINDS,
                     _with_weight, _describe, verbalize, _coerce_pair,
                     norm_lev, trapezoidal)
try:
    from fuzzy7 import Database7, _HAS_NL
except ImportError as _e:                                # pragma: no cover
    raise ImportError("fuzzy8 needs fuzzy7.py (which needs fuzzydb.py) "
                      "on the path") from _e

__version__ = "8.0"

# ===================== 1) clarification dialogue =========================

def _console_answer(word, options):
    print(f"  I don't know '{word}'. Did you mean:")
    for i, o in enumerate(options, 1):
        print(f"    {i}) {o['display']}   (match {o['score']:.2f})")
    print("    0) skip")
    try:
        raw = input("    choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if raw in ("", "0"):
        return None
    try:
        return options[int(raw) - 1]
    except (ValueError, IndexError):
        return None

class DialogueMixin:
    """Unknown words become clarification questions; each answer is saved
    into the NL lexicon, so the vocabulary grows with every conversation."""

    def _suggestions_for(self, table, word):
        cands = []
        for (t, c, lab) in sorted(self.fuzzy_sets):
            if t != table:
                continue
            s = norm_lev(word, lab)
            if s >= 0.68:
                cands.append({"kind": "pred", "value": f"{c} IS {lab}",
                              "display": f"{c} IS {lab}",
                              "score": round(s, 2)})
        seen = set()
        for w, col in self._col_words(table).items():
            if col in seen:
                continue
            s = norm_lev(word, w)
            if s >= 0.75:
                seen.add(col)
                cands.append({"kind": "column", "value": col,
                              "display": f"column {col}",
                              "score": round(s, 2)})
        for (tbl, w2), entry in self.synonyms.items():
            if tbl not in ("", table):
                continue
            s = norm_lev(word, w2)
            if s >= 0.75:
                tgt = (f"{entry['col']} IS {entry['label']}"
                       if entry["kind"] == "pred" else entry["text"])
                cands.append({"kind": "synonym", "value": w2,
                              "display": f"like '{w2}' -> {tgt}",
                              "score": round(s, 2)})
        cands.sort(key=lambda c: -c["score"])
        out, used = [], set()
        for c in cands:
            k = (c["kind"], c["value"])
            if k not in used:
                used.add(k)
                out.append(c)
        return out[:4]

    def clarify(self, table, text):
        """{interpretation, questions:[{word, options}]} for fixable unknowns."""
        if not hasattr(self, "interpret"):
            raise FuzzyError("clarification needs the NL layer (fuzzynl.py)")
        interp = self.interpret(table, text)
        questions = []
        for w in interp["unknown"]:
            opts = self._suggestions_for(table, w)
            if opts:
                questions.append({"word": w, "options": opts})
        return {"interpretation": interp, "questions": questions}

    def learn_from_answer(self, table, word, option):
        """Persist a clarification answer as permanent vocabulary."""
        word = str(word).lower().strip()
        if option["kind"] == "pred":
            self.define_synonym(word, option["value"])
        elif option["kind"] == "column":
            self.define_column_alias(word, option["value"])
        elif option["kind"] == "synonym":
            entry = self._synonym(table, option["value"])
            if entry is None:
                raise FuzzyError(f"synonym '{option['value']}' not found")
            if entry["kind"] == "pred":
                self.define_synonym(word,
                                    f"{entry['col']} IS {entry['label']}")
            else:
                self.define_synonym(word, entry["text"])
        else:
            raise FuzzyError(f"bad option kind {option['kind']!r}")
        return self

    def ask(self, table, text, answer_fn=None, max_rounds=6):
        """Interpret with clarification until unknowns resolve or are skipped.
        Returns {where, interpretation, learned:[{word, learned_as}]}."""
        if answer_fn is None:
            answer_fn = _console_answer
        learned = []
        for _ in range(max_rounds):
            cl = self.clarify(table, text)
            if not cl["questions"]:
                break
            q = cl["questions"][0]
            choice = answer_fn(q["word"], q["options"])
            if choice is None:
                break
            self.learn_from_answer(table, q["word"], choice)
            learned.append({"word": q["word"],
                            "learned_as": choice["display"]})
        interp = self.interpret(table, text)
        return {"where": interp["where"], "interpretation": interp,
                "learned": learned}

# ======================= 2) self-explanation =============================

class ExplainMixin:
    def _eval_where_row(self, table, where, row):
        ast = _parse_cached(where)
        leaves = collect_leaves(ast)
        degs = [self._eval_leaf(n, row, table) for n in leaves]
        lm = {id(n): d for n, d in zip(leaves, degs)}
        total = eval_tree(ast, lm, TNORMS["min"], SNORMS["max"],
                          self.quantifiers)
        return total, [_describe(n) for n in leaves], degs

    def _resolve_ref(self, table, ref):
        rows = [dict(r) for r in self.conn.execute(
            f"SELECT rowid AS __rid__, * FROM {self._ident(table)}")]
        if isinstance(ref, int) and not isinstance(ref, bool):
            for r in rows:
                if r["__rid__"] == ref:
                    return r
            raise FuzzyError(f"no rowid {ref} in {table}")
        if isinstance(ref, dict):
            for r in rows:
                if all(r.get(k) == v for k, v in ref.items()):
                    return r
            raise FuzzyError(f"no row in {table} matches {ref}")
        raise FuzzyError("ref must be a rowid or {col: value} matcher")

    def why_not(self, table, where, ref, target=1.0):
        """Why doesn't this row (fully) match? Per-predicate degrees, the
        single blocker, and counterfactual fixes via the v7 engine."""
        row = self._resolve_ref(table, ref)
        total, texts, degs = self._eval_where_row(table, where, row)
        ordered = sorted(zip(texts, degs), key=lambda p: p[1])
        blocker = None
        if ordered and ordered[0][1] < target - 1e-9:
            blocker = {"predicate": ordered[0][0],
                       "degree": round(ordered[0][1], 3)}
        cf = self.counterfactuals(table, where, row, target=target)
        return {"row": {k: v for k, v in row.items() if k != "__rid__"},
                "degree": round(total, 3), "verdict": verbalize(total),
                "predicate_degrees": {t: round(d, 3)
                                      for t, d in zip(texts, degs)},
                "blocker": blocker, "to_reach_target": cf}

    def which_queries_include(self, table, ref, alpha=0.5):
        """Inverse search: which saved queries / views would return this row?"""
        row = self._resolve_ref(table, ref)
        out = []
        for name, (tbl, w) in sorted(self.saved_queries.items()):
            if tbl.lower() != table.lower() or not w:
                continue
            try:
                deg, _, _ = self._eval_where_row(table, w, row)
            except FuzzyError:
                continue
            out.append({"source": "saved", "name": name, "where": w,
                        "degree": round(deg, 3), "included": deg >= alpha})
        for name, v in sorted(self.views.items()):
            if str(v.get("table", "")).lower() != table.lower():
                continue
            w = v.get("where")
            if not w:
                continue
            try:
                deg, _, _ = self._eval_where_row(table, w, row)
            except FuzzyError:
                continue
            out.append({"source": "view", "name": name, "where": w,
                        "degree": round(deg, 3), "included": deg >= alpha})
        out.sort(key=lambda x: -x["degree"])
        return out

# ================== 3) per-user preference profiles ======================

class ProfileMixin:
    def set_profile(self, user, table, weights):
        """weights: {predicate_text: weight}, e.g. learn_weights()['weights']."""
        user = str(user).lower()
        weights = {str(k): float(v) for k, v in weights.items()}
        self.profiles[user] = {"table": table.lower(), "weights": weights}
        if self._persistent:
            self.conn.execute(
                "INSERT OR REPLACE INTO _fz_profiles VALUES (?,?,?)",
                (user, table.lower(), json.dumps(weights)))
            self.conn.commit()
        return self

    def learn_for(self, user, table, where, liked=(), disliked=(), **kw):
        """learn_weights + store the result as user's profile."""
        res = self.learn_weights(table, where, liked=liked,
                                 disliked=disliked, **kw)
        self.set_profile(user, table, res["weights"])
        res["profile"] = user
        return res

    def get_profile(self, user):
        return self.profiles.get(str(user).lower())

    def personal_query(self, user, table, where=None, **kw):
        """Run a query with the user's learned predicate weights applied
        (aggregation='weighted'). Predicates the profile never saw keep
        weight 1.0, so new criteria still contribute."""
        prof = self.profiles.get(str(user).lower())
        if not prof or prof.get("table") != table.lower() or not where:
            return self.fuzzy_query(table, where, **kw)
        wmap = {k.lower(): float(v) for k, v in prof["weights"].items()}
        ast = _parse_cached(where)

        def applyw(node):
            kk = node[0]
            if kk in LEAF_KINDS:
                w = wmap.get(_describe(node).lower(), 1.0)
                return _with_weight(node, w)
            if kk in ("any", "all"):
                return (kk, [applyw(c) for c in node[1]])
            if kk == "quant":
                return (kk, node[1], [applyw(c) for c in node[2]])
            return (kk,) + tuple(applyw(c) for c in node[1:])

        wtext = _describe(applyw(ast))
        kw.setdefault("aggregation", "weighted")
        kw.setdefault("min_degree", 0.0)
        return self.fuzzy_query(table, wtext, **kw)

# ============================ 4) fuzzy AHP ===============================

_AHP_WORDS = {"equally": 1, "slightly": 2, "moderately": 3, "strongly": 5,
              "very_strongly": 7, "extremely": 9}
_AHP_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
           8: 1.41, 9: 1.45, 10: 1.49}

class AHPMixin:
    def ahp_weights(self, comparisons):
        """Saaty AHP. comparisons: (a, b, ratio) with ratio > 1 meaning a
        dominates b. ratio may be a number or a word from the Saaty scale
        (equally/slightly/moderately/strongly/very_strongly/extremely).
        Returns {name: weight, ..., '_consistency_ratio': cr} — warns when
        CR > 0.1. Feed into rank_alternatives(importance=...) or profiles."""
        names = []
        edges = []
        for comp in comparisons:
            a, b, r = comp
            if isinstance(r, str):
                rl = r.strip().lower().replace(" ", "_")
                if rl in _AHP_WORDS:
                    r = float(_AHP_WORDS[rl])
                else:
                    r = float(r)
            r = float(r)
            if r < 1:                              # accept reciprocals
                a, b, r = b, a, 1.0 / r
            for n in (a, b):
                if n not in names:
                    names.append(n)
            edges.append((a, b, r))
        n = len(names)
        if n < 2:
            raise FuzzyError("need comparisons over >= 2 criteria")
        idx = {nm: i for i, nm in enumerate(names)}
        M = [[1.0] * n for _ in range(n)]
        for a, b, r in edges:
            M[idx[a]][idx[b]] = r
            M[idx[b]][idx[a]] = 1.0 / r
        w = []
        for i in range(n):                          # geometric-mean priorities
            p = 1.0
            for j in range(n):
                p *= M[i][j]
            w.append(p ** (1.0 / n))
        s = sum(w) or 1.0
        w = [x / s for x in w]
        lam = sum(sum(M[i][j] * w[j] for j in range(n)) / w[i]
                  for i in range(n)) / n
        ci = (lam - n) / (n - 1) if n > 1 else 0.0
        cr = ci / _AHP_RI.get(n, 1.49) if n > 2 else 0.0
        out = {names[i]: round(w[i], 4) for i in range(n)}
        out["_consistency_ratio"] = round(cr, 3)
        if cr > 0.1:
            out["_warning"] = ("CR > 0.1 — pairwise comparisons are "
                               "inconsistent; revise them")
        return out

# ============================ 5) alerts ==================================

_ALERT_KINDS = ("constraint", "drift", "count", "aggregate")

class AlertMixin:
    """Operational watchdogs over the fuzzy layer. run_checks() evaluates
    every alert and returns the fired ones; watch() loops. FuzzyErrors
    inside a check (e.g. vocabulary not yet defined) skip that alert."""

    def define_alert(self, name, kind, **params):
        kind = str(kind).lower()
        if kind not in _ALERT_KINDS:
            raise FuzzyError(f"unknown alert kind {kind!r} "
                             f"(have: {list(_ALERT_KINDS)})")
        if kind == "aggregate" and not params.get("column"):
            raise FuzzyError("aggregate alerts need a 'column'")
        self.alerts[name] = {"kind": kind, "params": params}
        if self._persistent:
            self.conn.execute("INSERT OR REPLACE INTO _fz_alerts "
                              "VALUES (?,?,?)",
                              (name, kind, json.dumps(params)))
            self.conn.commit()
        return self

    def drop_alert(self, name):
        self.alerts.pop(name, None)
        if self._persistent:
            self.conn.execute("DELETE FROM _fz_alerts WHERE name=?", (name,))
            self.conn.commit()
        return self

    def run_checks(self):
        fired = []
        for name, a in sorted(self.alerts.items()):
            k, p = a["kind"], a["params"]
            try:
                if k == "constraint":
                    v = self.check_table(p["table"])
                    if len(v) > p.get("max_violations", 0):
                        fired.append({"alert": name, "kind": k,
                                      "severity": "high",
                                      "detail": f"{len(v)} violating row(s) "
                                                f"in {p['table']}"})
                elif k == "drift":
                    ds = self.detect_drift(p["table_a"], p["table_b"],
                                           columns=p.get("columns"))
                    worst = max((d["drift"] for d in ds), default=0.0)
                    if ds and worst >= p.get("min_drift", 0.2):
                        top = ds[0]
                        fired.append({"alert": name, "kind": k,
                                      "severity":
                                          "high" if worst >= 0.3 else "medium",
                                      "detail": f"drift {worst:.2f} on "
                                                f"{top['column']}."
                                                f"{top['label']}"})
                elif k == "count":
                    res = self.fuzzy_query(p["table"], p.get("where"),
                                           min_degree=p.get("alpha", 0.5))
                    n_ = len(res)
                    lo, hi = p.get("below"), p.get("above")
                    if (lo is not None and n_ < lo) or \
                            (hi is not None and n_ > hi):
                        fired.append({"alert": name, "kind": k,
                                      "severity": "medium",
                                      "detail": f"{n_} row(s) match "
                                                f"{p.get('where') or 'ALL'}"})
                elif k == "aggregate":
                    agg = self.fuzzy_aggregate(p["table"], p.get("where"),
                                               agg_column=p["column"],
                                               func=p.get("func", "mean"))
                    val = agg.get("value")
                    lo, hi = p.get("below"), p.get("above")
                    if val is not None and \
                            ((lo is not None and val < lo) or
                             (hi is not None and val > hi)):
                        fired.append({"alert": name, "kind": k,
                                      "severity": "medium",
                                      "detail":
                                          f"{p.get('func', 'mean')}"
                                          f"({p['column']}) = {val}"})
            except FuzzyError:
                continue
        return fired

    def watch(self, interval=60.0, callback=None, once=False):
        """Poll run_checks(); callback(fires) per round. once=True returns
        the fires of a single round (handy for cron). Ctrl-C stops."""
        import time
        if callback is None:
            def callback(fires):
                for f in fires:
                    print(f"[ALERT] {f['alert']} ({f['kind']}): {f['detail']}")
        while True:
            fires = self.run_checks()
            if fires:
                callback(fires)
            if once:
                return fires
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                return None

# ========================== 6) time travel ===============================

class TimeMixin:
    """Snapshot-based versioning: cheap CREATE TABLE AS SELECT copies that
    the CURRENT vocabulary can still query — so you can ask 'who was high
    back then?' with today's (or any) definitions."""

    def snapshot(self, table, name=None):
        table = str(table).lower()
        name = (name or _dt.datetime.now().strftime(
            f"{table}_%Y%m%d_%H%M%S")).lower()
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise FuzzyError(f"bad snapshot name {name!r}")
        snap = "_snap_" + name
        try:
            self.conn.execute(f"CREATE TABLE {self._ident(snap)} "
                              f"AS SELECT * FROM {self._ident(table)}")
        except sqlite3.OperationalError as e:
            raise FuzzyError(f"snapshot failed: {e}")
        created = _dt.datetime.now().isoformat(timespec="seconds")
        self.snapshots[name] = {"table": table, "snap": snap,
                                "created": created}
        if self._persistent:
            self.conn.execute("INSERT OR REPLACE INTO _fz_snapshots "
                              "VALUES (?,?,?,?)",
                              (name, table, snap, created))
            self.conn.commit()
        return name

    def list_snapshots(self, table=None):
        return {n: dict(v) for n, v in sorted(self.snapshots.items())
                if table is None or v["table"] == table.lower()}

    def _snap_info(self, table, as_of):
        info = self.snapshots.get(as_of)
        if info is None or info["table"] != table.lower():
            avail = sorted(n for n, i in self.snapshots.items()
                           if i["table"] == table.lower())
            raise FuzzyError(f"no snapshot {as_of!r} for {table!r}; "
                             f"have: {avail}")
        return info

    def as_of_query(self, table, where=None, as_of=None, **kw):
        """fuzzy_query against a snapshot, evaluated with the CURRENT
        vocabulary (MFs are looked up under the original table name)."""
        table = str(table).lower()
        if as_of in (None, "", "now"):
            return self.fuzzy_query(table, where, **kw)
        info = self._snap_info(table, as_of)
        ast = _parse_cached(where) if where else None
        leaves = collect_leaves(ast) if ast else []
        rows = [dict(r) for r in self.conn.execute(
            f"SELECT * FROM {self._ident(info['snap'])}")]
        out = []
        for row in rows:
            degs = [self._eval_leaf(n, row, table) for n in leaves]
            if not degs:
                deg = 1.0
            else:
                lm = {id(n): d for n, d in zip(leaves, degs)}
                deg = eval_tree(ast, lm, TNORMS["min"], SNORMS["max"],
                                self.quantifiers)
            if deg >= kw.get("min_degree", 0.0):
                out.append((deg, row))
        out.sort(key=lambda p: -p[0])
        if kw.get("limit"):
            out = out[:kw["limit"]]
        return out

    def diff_as_of(self, table, as_of):
        """Rows added / removed / changed (by rowid) since the snapshot."""
        table = str(table).lower()
        info = self._snap_info(table, as_of)
        old = {r["rowid"]: dict(r) for r in self.conn.execute(
            f"SELECT rowid, * FROM {self._ident(info['snap'])}")}
        new = {r["rowid"]: dict(r) for r in self.conn.execute(
            f"SELECT rowid, * FROM {self._ident(table)}")}
        added = [new[i] for i in sorted(set(new) - set(old))]
        removed = [old[i] for i in sorted(set(old) - set(new))]
        changed = []
        for i in sorted(set(new) & set(old)):
            diffs = {k: [old[i].get(k), new[i].get(k)]
                     for k in new[i] if k != "rowid"
                     and old[i].get(k) != new[i].get(k)}
            if diffs:
                changed.append({"rowid": i, "changes": diffs})
        return {"snapshot": as_of, "added": added, "removed": removed,
                "changed": changed}

# ======================== 7) pandas interop ==============================

class InteropMixin:
    def to_dataframe(self, results):
        """fuzzy_query results -> pandas DataFrame (degree + row columns,
        explain dicts as _prefixed columns)."""
        try:
            import pandas as pd
        except ImportError:
            raise FuzzyError("pandas is not installed: pip install pandas")
        rows = []
        for item in results:
            d = {"degree": round(item[0], 4)}
            d.update(dict(item[1]))
            if len(item) == 3 and isinstance(item[2], dict):
                d.update({f"_{k}": v for k, v in item[2].items()})
            rows.append(d)
        return pd.DataFrame(rows)

    def from_dataframe(self, df, table):
        """DataFrame -> table (numeric dtypes become REAL, NaN -> NULL)."""
        spec = {}
        for c in df.columns:
            nm = self._clean_header(c)
            dt = str(df[c].dtype).lower()
            spec[nm] = "REAL" if ("int" in dt or "float" in dt) else "TEXT"
        self.create_table(table, **spec)
        recs = []
        for rec in df.to_dict("records"):
            row = {}
            for c in df.columns:
                v = rec[c]
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    v = None
                row[self._clean_header(c)] = v
            recs.append(row)
        self.insert_many(table, recs)
        return len(recs)

# ====================== 8) web UI + HTTP server ==========================

_UI = r"""<!doctype html><html><head><meta charset="utf-8">
<title>fuzzydb console</title><style>
body{font-family:system-ui,sans-serif;margin:24px;max-width:1080px;background:#fafafa;color:#222}
h2{margin-top:0}
textarea{width:100%;box-sizing:border-box;font-size:15px;padding:8px}
button{padding:6px 14px;font-size:15px;cursor:pointer}
table{border-collapse:collapse;margin-top:14px;width:100%}
td,th{border:1px solid #e2e2e2;padding:6px 10px;text-align:left;font-size:14px}
th{background:#efefef}
#interp{background:#eef6ee;border:1px solid #cde3cd;padding:10px;margin-top:10px;white-space:pre-wrap;display:none}
#why{background:#fdf3e7;border:1px solid #ecd9b0;padding:10px;margin-top:10px;white-space:pre-wrap;display:none}
.whybtn{color:#07c;background:none;border:none;text-decoration:underline;padding:0;font-size:13px}
#status{color:#666;margin-left:10px}
</style></head><body>
<h2>fuzzydb console</h2>
<div>table <select id="tbl"></select>
 alpha <input id="alpha" value="0.05" size="4">
 limit <input id="limit" value="15" size="4">
 <button onclick="loadTables()">reload</button></div>
<p><textarea id="q" rows="3" placeholder="age IS young AND salary IS high&#10;&#10;...or plain English: young people who earn a lot"></textarea></p>
<div><button onclick="run()">Run</button><span id="status"></span></div>
<div id="interp"></div><table id="res"></table><div id="why"></div>
<script>
const $=i=>document.getElementById(i);
let rows=[];
async function j(u,o){const r=await fetch(u,o);return r.json();}
async function loadTables(){const d=await j('/api/tables');
  $('tbl').innerHTML=(d.tables||[]).map(t=>'<option>'+t+'</option>').join('');}
function color(d){const h=120-Math.max(0,Math.min(1,d))*120;
  return 'hsl('+h+',70%,'+(93-20*d)+'%)';}
async function run(){
  $('status').textContent='...';$('res').innerHTML='';$('why').style.display='none';
  const d=await j('/api/query',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({table:$('tbl').value,where:$('q').value,
      min_degree:parseFloat($('alpha').value)||0,
      limit:parseInt($('limit').value)||15,explain:true})});
  if(d.error){$('status').textContent='error: '+d.error;return;}
  $('status').textContent=(d.results||[]).length+' rows';
  const ip=$('interp');
  if(d.interpreted_as){ip.style.display='block';
    ip.textContent='interpreted as: '+d.interpreted_as+
      (d.confidence!=null?'   (confidence '+Math.round(d.confidence*100)+'%)':'')+
      (d.unknown&&d.unknown.length?'\nunrecognized: '+d.unknown.join(', '):'');}
  else ip.style.display='none';
  rows=(d.results||[]).map(r=>r.row);
  if(!d.results||!d.results.length)return;
  const keys=Object.keys(d.results[0].row);
  let h='<tr><th>deg</th><th></th>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr>';
  d.results.forEach((r,i)=>{
    h+='<tr style="background:'+color(r.degree)+'"><td><b>'+r.degree.toFixed(2)+'</b></td>'+
       '<td><button class="whybtn" onclick="why('+i+')">why?</button></td>'+
       keys.map(k=>'<td>'+(r.row[k]??'')+'</td>').join('')+'</tr>';});
  $('res').innerHTML=h;
}
async function why(i){
  const d=await j('/api/why',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({table:$('tbl').value,where:$('q').value,row:rows[i]})});
  const w=$('why');
  if(d.error){w.style.display='block';w.textContent='error: '+d.error;return;}
  w.style.display='block';
  let s='degree '+d.degree+' - '+d.verdict+'\n';
  for(const[p,v]of Object.entries(d.predicate_degrees))
    s+='  '+v.toFixed(2)+'  '+p+'\n';
  if(d.blocker)s+='blocker: '+d.blocker.predicate+' ('+d.blocker.degree+')\n';
  ((d.to_reach_target&&d.to_reach_target.suggestions)||[]).slice(0,3)
    .forEach(x=>s+='-> '+x.suggestion+'\n');
  w.textContent=s;
}
loadTables();
</script></body></html>"""

def serve8(path, host="127.0.0.1", port=8200):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse
    db = Database8(path, check_same_thread=False)

    class H(BaseHTTPRequestHandler):
        def _j(self, code, obj):
            data = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _html(self, text):
            data = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                with db._lock:
                    if u.path == "/":
                        self._html(_UI)
                    elif u.path == "/api/tables":
                        self._j(200, {"tables": db.tables()})
                    else:
                        self._j(404, {"error": "not found"})
            except Exception as e:
                self._j(400, {"error": str(e)})

        @staticmethod
        def _query(body):
            table = body["table"]
            where = (body.get("where") or "").strip()
            kw = {"min_degree": float(body.get("min_degree") or 0),
                  "limit": int(body.get("limit") or 15)}
            if body.get("explain"):
                kw["explain"] = True
            if not where:
                res = db.fuzzy_query(table, None, **kw)
                return {"results": [{"degree": r[0], "row": r[1]}
                                    for r in res]}
            try:
                _parse_cached(where)
                fuzzy = True
            except FuzzySyntaxError:
                fuzzy = False
            if fuzzy:
                res = db.fuzzy_query(table, where, **kw)
                return {"interpreted_as": where,
                        "results": [{"degree": r[0], "row": r[1],
                                     **({"why": r[2]} if len(r) == 3 else {})}
                                    for r in res]}
            r = db.nl_query(table, where, **kw)       # plain English path
            itp = r.get("interpretation") or {}
            return {"interpreted_as": r.get("interpreted_as"),
                    "results": [{"degree": x[0], "row": x[1]}
                                for x in r.get("results", [])],
                    "confidence": itp.get("confidence"),
                    "unknown": itp.get("unknown")}

        @staticmethod
        def _why(body):
            table = body["table"]
            where = (body.get("where") or "").strip()
            ref = body.get("rowid") or body.get("row")
            if not ref:
                raise FuzzyError("need 'rowid' or 'row'")
            try:
                _parse_cached(where)
            except FuzzySyntaxError:
                where = db.interpret(table, where)["where"]
                if not where:
                    raise FuzzyError("could not interpret 'where'")
            return db.why_not(table, where, ref)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                with db._lock:
                    if u.path == "/api/query":
                        self._j(200, self._query(body))
                    elif u.path == "/api/why":
                        self._j(200, self._why(body))
                    else:
                        self._j(404, {"error": "not found"})
            except Exception as e:
                self._j(400, {"error": str(e)})

        def log_message(self, *a):
            pass

    print(f"fuzzydb console on http://{host}:{port}  — the query box takes "
          f"fuzzy WHERE or plain English; 'why?' explains any row")
    try:
        ThreadingHTTPServer((host, port), H).serve_forever()
    except KeyboardInterrupt:
        db.close()

# ====================== assemble the database class ======================

class Database8(DialogueMixin, ExplainMixin, ProfileMixin, AHPMixin,
                AlertMixin, TimeMixin, InteropMixin, Database7):
    META_TABLES = Database7.META_TABLES | {"_fz_lexicon", "_fz_profiles",
                                           "_fz_alerts", "_fz_snapshots"}

    def __init__(self, path=":memory:", **kw):
        super().__init__(path, **kw)
        self.profiles = {}
        self.alerts = {}
        self.snapshots = {}
        if self._persistent:
            self._install_meta8()
            self._load_meta8()

    def tables(self):
        return [t for t in super().tables() if not t.startswith("_snap_")]

    def _install_meta8(self):
        c = self.conn
        c.execute("CREATE TABLE IF NOT EXISTS _fz_profiles (user TEXT "
                  "PRIMARY KEY, tbl TEXT, weights TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS _fz_alerts (name TEXT "
                  "PRIMARY KEY, kind TEXT, params TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS _fz_snapshots (name TEXT "
                  "PRIMARY KEY, tbl TEXT, snap TEXT, created TEXT)")
        c.commit()

    def _load_meta8(self):
        for user, tbl, wj in self.conn.execute(
                "SELECT user, tbl, weights FROM _fz_profiles"):
            try:
                self.profiles[user] = {"table": tbl,
                                       "weights": json.loads(wj)}
            except ValueError:
                pass
        for name, kind, pj in self.conn.execute(
                "SELECT name, kind, params FROM _fz_alerts"):
            try:
                self.alerts[name] = {"kind": kind, "params": json.loads(pj)}
            except ValueError:
                pass
        for name, tbl, snap, created in self.conn.execute(
                "SELECT name, tbl, snap, created FROM _fz_snapshots"):
            self.snapshots[name] = {"table": tbl, "snap": snap,
                                    "created": created}

Database = Database8        # friendly alias

# ============================== self test ================================

def selftest():
    fails, total = [], [0]

    def ok(cond, name):
        total[0] += 1
        print(("  PASS  " if cond else "  FAIL  ") + name)
        if not cond:
            fails.append(name)

    db = Database8(":memory:")
    db.create_table("employees", name="TEXT", age="REAL", salary="REAL",
                    experience="REAL")
    db.insert_many("employees", [
        {"name": "Alice", "age": 26, "salary": 72000, "experience": 3},
        {"name": "Bob", "age": 34, "salary": 85000, "experience": 9},
        {"name": "Cynthia", "age": 41, "salary": 118000, "experience": 17},
        {"name": "Dave", "age": 52, "salary": 64000, "experience": 24},
        {"name": "Elena", "age": 29, "salary": 94000, "experience": 6},
        {"name": "Frank", "age": 47, "salary": 132000, "experience": 21},
        {"name": "Grace", "age": 23, "salary": 51000, "experience": 1},
        {"name": "Hassan", "age": 38, "salary": 76000, "experience": 12},
    ])
    db.define_fuzzy_set("employees", "age", "young", trapezoidal(0, 0, 26, 38))
    db.define_fuzzy_set("employees", "age", "old",
                        trapezoidal(40, 52, 120, 120))
    db.define_fuzzy_set("employees", "salary", "high",
                        trapezoidal(80000, 100000, float("inf"),
                                    float("inf")))
    db.define_fuzzy_set("employees", "salary", "extreme",
                        trapezoidal(120000, 130000, float("inf"),
                                    float("inf")))
    db.define_fuzzy_set("employees", "experience", "seasoned",
                        trapezoidal(10, 18, float("inf"), float("inf")))
    db.autofuzz("employees", "experience", 3)

    # --- AHP ---
    w = db.ahp_weights([("salary", "age", 5), ("salary", "experience", 3),
                        ("experience", "age", 5.0 / 3.0)])
    ok(w["salary"] > w["experience"] > w["age"] and
       w["_consistency_ratio"] < 0.05, "AHP priorities + consistency")
    w2 = db.ahp_weights([("a", "b", "strongly")])
    ok(abs(w2["a"] - 5.0 / 6.0) < 1e-6, "AHP word scale + reciprocals")

    # --- dialogue ---
    if _HAS_NL:
        r = db.ask("employees", "very seasond folks",
                   answer_fn=lambda wd, o: o[0] if o else None)
        ok(r["where"] == "experience IS very seasoned",
           f"dialogue teaches -> {r['where']}")
        ok(db._synonym("employees", "seasond") is not None,
           "learned synonym registered")
    else:
        print("  SKIP  dialogue (no fuzzynl)")

    # --- why_not / which_queries_include ---
    why = db.why_not("employees", "age IS young AND salary IS high",
                     {"name": "Grace"})
    ok(why["degree"] == 0.0 and
       why["blocker"]["predicate"] == "salary IS high", "why_not blocker")
    ok(abs(why["to_reach_target"]["suggestions"][0]["value"] - 100000) < 1,
       "why_not counterfactual value")
    db.save_query("employees", "young_high", "age IS young AND salary IS high")
    inc_e = db.which_queries_include("employees", {"name": "Elena"})
    inc_g = db.which_queries_include("employees", {"name": "Grace"})
    ok(any(x["name"] == "young_high" and x["included"] for x in inc_e),
       "Elena included by saved query")
    ok(all(not x["included"] for x in inc_g), "Grace excluded everywhere")

    # --- profiles ---
    db.learn_for("recruiter", "employees", "age IS young AND salary IS high",
                 liked=[{"name": "Elena"}, {"name": "Bob"}],
                 disliked=[{"name": "Cynthia"}, {"name": "Frank"}])
    base = {r_[1]["name"]: r_[0] for r_ in db.fuzzy_query(
        "employees", "age IS young AND salary IS high")}
    pers = {r_[1]["name"]: r_[0] for r_ in db.personal_query(
        "recruiter", "employees", "age IS young AND salary IS high")}
    ok(pers["Grace"] > base["Grace"] and pers["Grace"] >= 0.99,
       "personalization boosts Grace")
    ok(pers["Cynthia"] < 0.05, "personalization zeroes Cynthia")

    # --- time travel ---
    db.snapshot("employees", "before_zoe")
    db.insert("employees", name="Zoe", age=79, salary=250000, experience=2)
    old = db.as_of_query("employees", "salary IS high",
                         as_of="before_zoe", min_degree=0.5)
    new = db.fuzzy_query("employees", "salary IS high", min_degree=0.5)
    ok(len(old) == 3 and len(new) == 4,
       f"time travel {len(old)} rows as-of -> {len(new)} now")
    d = db.diff_as_of("employees", "before_zoe")
    ok(len(d["added"]) == 1 and d["added"][0]["name"] == "Zoe",
       "diff_as_of detects the new row")

    # --- alerts ---
    db.define_alert("hc", "count", table="employees",
                    where="salary IS high", above=2)
    db.define_alert("ext", "constraint", table="employees",
                    max_violations=0)
    fires = db.run_checks()
    ok({"hc", "ext"} <= {f["alert"] for f in fires}, "alerts fire")

    # --- pandas (optional) ---
    try:
        import pandas  # noqa: F401
        have_pd = True
    except ImportError:
        have_pd = False
    if have_pd:
        df = db.to_dataframe(db.fuzzy_query("employees", "salary IS high",
                                            min_degree=0.5, limit=3))
        ok(len(df) == 3 and "degree" in df.columns, "to_dataframe")
        db2 = Database8(":memory:")
        n = db2.from_dataframe(df, "people")
        ok(n == 3 and db2.tables() == ["people"], "from_dataframe")
    else:
        print("  SKIP  pandas not installed")

    # --- persistence ---
    path = tempfile.mktemp(suffix=".db")
    try:
        with Database8(path) as f1:
            f1.create_table("t", v="REAL")
            f1.insert("t", v=1.0)
            f1.set_profile("u1", "t", {"v IS high": 0.7})
            f1.define_alert("a1", "count", table="t", above=0)
            f1.snapshot("t", "s1")
        with Database8(path) as f2:
            ok(f2.get_profile("u1")["weights"] == {"v IS high": 0.7},
               "profile persisted")
            ok("a1" in f2.alerts and "s1" in f2.snapshots,
               "alerts + snapshots persisted")
            ok(len(f2.as_of_query("t", None, as_of="s1")) == 1,
               "as_of after reopen")
    finally:
        if os.path.exists(path):
            os.unlink(path)

    print(f"\n  {total[0] - len(fails)}/{total[0]} passed")
    return len(fails)

# ================================ demo ===================================

def demo():
    db = Database8(":memory:")
    db.create_table("employees", name="TEXT", age="REAL", salary="REAL",
                    experience="REAL")
    db.insert_many("employees", [
        {"name": "Alice", "age": 26, "salary": 72000, "experience": 3},
        {"name": "Bob", "age": 34, "salary": 85000, "experience": 9},
        {"name": "Cynthia", "age": 41, "salary": 118000, "experience": 17},
        {"name": "Dave", "age": 52, "salary": 64000, "experience": 24},
        {"name": "Elena", "age": 29, "salary": 94000, "experience": 6},
        {"name": "Frank", "age": 47, "salary": 132000, "experience": 21},
        {"name": "Grace", "age": 23, "salary": 51000, "experience": 1},
        {"name": "Hassan", "age": 38, "salary": 76000, "experience": 12},
    ])
    db.define_fuzzy_set("employees", "age", "young", trapezoidal(0, 0, 26, 38))
    db.define_fuzzy_set("employees", "age", "old",
                        trapezoidal(40, 52, 120, 120))
    db.define_fuzzy_set("employees", "salary", "high",
                        trapezoidal(80000, 100000, float("inf"),
                                    float("inf")))
    db.define_fuzzy_set("employees", "salary", "extreme",
                        trapezoidal(120000, 130000, float("inf"),
                                    float("inf")))
    db.define_fuzzy_set("employees", "experience", "seasoned",
                        trapezoidal(10, 18, float("inf"), float("inf")))
    db.autofuzz("employees", "experience", 3)

    print("=" * 68)
    print("1) CLARIFICATION DIALOGUE (scripted user answers)")
    if _HAS_NL:
        def scripted(word, options):
            print(f"   ? '{word}' -> did you mean: " +
                  " | ".join(o["display"] for o in options))
            return options[0] if options else None

        r = db.ask("employees", "find me very seasond folks",
                   answer_fn=scripted)
        print("   learned    :", [(l["word"], l["learned_as"])
                                  for l in r["learned"]])
        print("   interpreted:", r["where"])
        print("   (persisted — the typo is now permanent vocabulary)")
    else:
        print("   (fuzzynl.py not present — skipped)")

    print("\n" + "=" * 68)
    print("2) WHY NOT? + WHICH QUERIES INCLUDE ME?")
    db.save_query("employees", "young_high", "age IS young AND salary IS high")
    where = "age IS young AND salary IS high"
    for nm in ("Grace", "Elena"):
        w = db.why_not("employees", where, {"name": nm})
        print(f"\n   {nm}: degree {w['degree']} — {w['verdict']}")
        for p, v in w["predicate_degrees"].items():
            print(f"      {v:.2f}  {p}")
        if w["blocker"]:
            print(f"      blocker: {w['blocker']['predicate']}")
        for s in w["to_reach_target"]["suggestions"][:1]:
            print(f"      -> {s['suggestion']}")
        inc = db.which_queries_include("employees", {"name": nm})
        hits = [x for x in inc if x["included"]]
        print("      included by:",
              ", ".join(f"{x['source']}:{x['name']} ({x['degree']})"
                        for x in hits) or "(nothing)")

    print("\n" + "=" * 68)
    print("3) PER-USER PROFILES")
    res = db.learn_for("recruiter", "employees",
                       "age IS young AND salary IS high",
                       liked=[{"name": "Elena"}, {"name": "Bob"}],
                       disliked=[{"name": "Cynthia"}, {"name": "Frank"}])
    print("   learned weights:", res["weights"])
    base = {r_[1]["name"]: r_[0] for r_ in db.fuzzy_query(
        "employees", "age IS young AND salary IS high")}
    pers = {r_[1]["name"]: r_[0] for r_ in db.personal_query(
        "recruiter", "employees", "age IS young AND salary IS high")}
    print(f"   {'name':<10}{'base':>7}{'personal':>11}")
    for nm in ("Grace", "Elena", "Bob", "Cynthia"):
        print(f"   {nm:<10}{base[nm]:>7.2f}{pers[nm]:>11.2f}")

    print("\n" + "=" * 68)
    print("4) AHP: WEIGHTS FROM PAIRWISE COMPARISONS")
    w = db.ahp_weights([("salary", "age", "strongly"),
                        ("salary", "experience", "moderately"),
                        ("experience", "age", "slightly")])
    for k, v in w.items():
        if not k.startswith("_"):
            print(f"   {k:<12} {v}")
    print("   CR =", w["_consistency_ratio"], w.get("_warning", ""))
    wc = db.ahp_weights([("salary", "age", 5),
                         ("salary", "experience", 3),
                         ("experience", "age", 5.0 / 3.0)])
    print("   consistent set CR:", wc["_consistency_ratio"])
    rank = db.rank_alternatives(
        "employees", {"salary": "high", "experience": "seasoned"},
        importance={"salary": wc["salary"],
                    "experience": wc["experience"]}, limit=3)
    print("   TOPSIS with AHP weights:",
          ", ".join(f"{r_[1]['name']} {r_[0]:.2f}" for r_ in rank))

    print("\n" + "=" * 68)
    print("5) TIME TRAVEL")
    db.snapshot("employees", "before_zoe")
    print("   snapshot 'before_zoe' taken",
          f"({len(db.as_of_query('employees', as_of='before_zoe'))} rows)")
    db.insert("employees", name="Zoe", age=79, salary=250000, experience=2)
    oldn = len(db.as_of_query("employees", "salary IS high",
                              as_of="before_zoe", min_degree=0.5))
    newn = len(db.fuzzy_query("employees", "salary IS high",
                              min_degree=0.5))
    print(f"   'salary IS high' (alpha 0.5): as-of {oldn} rows -> "
          f"now {newn} rows")
    d = db.diff_as_of("employees", "before_zoe")
    print("   diff:", [r_["name"] for r_ in d["added"]], "added,",
          len(d["removed"]), "removed,", len(d["changed"]), "changed")

    print("\n" + "=" * 68)
    print("6) ALERTS (watchdog)")
    db.define_alert("high_pay_headcount", "count", table="employees",
                    where="salary IS high", above=2)
    db.define_alert("salary_extreme", "constraint", table="employees",
                    max_violations=0)

    def bell(fires):
        for f in fires:
            print(f"   [ALERT] {f['alert']} ({f['kind']}, {f['severity']}): "
                  f"{f['detail']}")

    db.watch(once=True, callback=bell)
    print("   (db.watch(interval=60) would keep polling)")

    print("\n" + "=" * 68)
    print("7) PANDAS (optional)")
    try:
        import pandas  # noqa: F401
        df = db.to_dataframe(db.fuzzy_query("employees", "salary IS high",
                                            min_degree=0.5, limit=3))
        print("   DataFrame:", df.shape, "columns:",
              list(df.columns)[:4], "...")
    except FuzzyError as e:
        print("   ", e)

    print("\nWeb console:  python fuzzy8.py serve mydata.db 8200")
    print("Tests:        python fuzzy8.py selftest")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        demo()
    elif args[0] == "selftest":
        sys.exit(1 if selftest() else 0)
    elif args[0] == "serve":
        serve8(args[1] if len(args) > 1 else "fuzzy.db",
               port=int(args[2]) if len(args) > 2 else 8200)
    else:
        print("usage: python fuzzy8.py [demo | selftest | serve f.db port]")