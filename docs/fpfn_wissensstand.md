# Wissensstand: FP/FN als Signal für Latent-TTA (Stand 2026-11-20)

Kompakte Zusammenfassung aller Experimente, Ergebnisse und offenen Punkte zum FP/FN-Ansatz, damit
ein neuer Chat ohne den alten Verlauf weiterarbeiten kann. Ausführliche Abbildungen:
[src/notebooks/fpfn_findings.ipynb](../src/notebooks/fpfn_findings.ipynb) (Deutsch, alle Zahlen
werden dort aus `results/` und `logs/` berechnet). Branch: `fpfn-only`.

## 1. Kontext und Setup

- **Ziel:** Test-Time-Adaptation (TTA) der Prostata-Segmentierung (PMRI) unter Domain-Shift, gesteuert
  durch eine Qualitätsschätzung statt Labels.
- **TTA:** pro Bild/Batch eine kanalweise affine Transformation (γ, β) auf U-Net-Features an den
  Stellen `adn` [0, 17], nach der InstanceNorm (`tta.placement=post`), Adam lr 0.01, 100 Schritte.
  Code: `run_latent_tta` in [src/model/latent_adapter.py](../src/model/latent_adapter.py), Einstieg
  [src/apps/eval_latent_tta.sh](../src/apps/eval_latent_tta.sh) /
  [eval_latent_tta.py](../src/apps/eval_latent_tta.py).
- **Score-Predictor μ̂/κ** (Beta-Verteilung über Dice): `pre-trained/score-predictors/pre_norm_with_logits/`,
  `cleanHalf_0–4`, liest Swivels `first-embedding-last` = Layer 0, 17 und die Logits.
- **FP/FN-Head:** gleiche Architektur und Swivels wie μ̂, sagt log(1+FP), log(1+FN) vorher (MSE).
  Code [src/model/fpfn_head.py](../src/model/fpfn_head.py), Training
  [src/apps/train_fpfn_head.sh](../src/apps/train_fpfn_head.sh), Checkpoints
  `pre-trained/fpfn-heads/`, Auswertung ohne TTA `results/fpfn_head/<ckpt>/eval_data.pkl`.
- **Splits:** 3T→1.5T (Hauptsplit, alle Head-Experimente nur hier), PROMISE12, 1.5T→3T.
  Dice ohne TTA (Test): 0.677 / 0.642 / 0.526.
- **Vorgabe aus dem Meeting 2026-11-10:** μ̂ und κ weglassen, nur FP/FN, Perturbationen im Eingaberaum.

## 2. Ergebnisse

### 2.1 Orakel mit echten FP/FN (Labels, nicht einsetzbar)

Test-Dice nach 100 Schritten (3T→1.5T / PROMISE12 / 1.5T→3T):

| Variante | Dice |
|---|---|
| μ̂ allein | 0.753 / 0.702 / 0.538 |
| Größen-Orakel (Zielgröße aus echten FP₀/FN₀, einmal) | 0.783 / 0.742 / 0.590 (±30 % Rauschen egal) |
| μ̂ + log(1+FP) + log(1+FN) („getrennt“) | 0.707 / 0.673 / 0.562 — entfernt FP, lässt FN wachsen |
| μ̂ + log(1+FP+FN) („gemeinsam“) | 0.889 / 0.875 / 0.677 |
| nur log(1+FP+FN), ohne μ̂ (exp8) | 0.886 (nur 3T→1.5T) |
| Dice-Orakel | 0.937 / 0.926 / 0.874 |

- Weiche Zählung FP = Σp(1−g), FN = Σ(1−p)g. **Ihr Gradient nach p ist die Label-Karte** — er sagt
  jedem Pixel, wohin es muss. Darum wirkt das Orakel, nicht wegen der Zahl selbst.
- Nur die gemeinsame Form funktioniert; mit nur einem Term ist Adam invariant gegen dessen Gewicht.
- **Verpasste Prostatas** (Dice < 0.1 vor TTA: 12 / 9 / 25 % der Bilder): FP/FN-Orakel dort nur
  0.41 / 0.33 / 0.10 (Dice-Orakel 0.83 / 0.80 / 0.71); auf den übrigen Bildern so gut wie das
  Dice-Orakel. Lokales Minimum ist die leere Maske (4–7 % gemeinsam, bis 15 % getrennt geleert; alle
  hatten vorher Dice ≈ 0).
- Läufe ohne μ̂ und ohne Head sind für alle 5 `cleanHalf_k` identisch (gleiche U-Net) → effektiv ein Lauf.

### 2.2 FP/FN-Heads (ohne TTA)

Head-Varianten (alle 3T→1.5T, je 5 Seeds; Adversary verschiebt Eingabebilder so, dass der Head mehr
Fehler vorhersagt, Controller auf echten Dice-Abfall 0.1; 30 saubere + 30 Adversary-Bilder pro Schritt):

| Name | Zielgröße | Adversary |
|---|---|---|
| `fpfn` (hart) | harte Maske | einer, +∇ log(1+FP̂+FN̂) |
| `fpfnsoft` | weiche Zählung | wie hart |
| `fpfnsoftdown` | weich | + zweiter Adversary in TTA-Richtung (−∇, Vorhersage −0.5 in log) |
| `fpfnsplit` | hart | 15 Bilder +∇FP̂, 15 +∇FN̂, eigene Controller; Loss weiter über FP und FN |
| `fpfnsepfp` + `fpfnsepfn` | hart | zwei Netze, je nur eine Zahl + eigener Adversary; 200 Epochen |

Pearson r von log(1+Anzahl), Mittel über 5 Seeds (Training / Validierung / Test):

| Head | FP | FN | FN̂/FN Test |
|---|---|---|---|
| hart | 0.74 / −0.01 / 0.24 | 0.87 / 0.20 / 0.44 | 0.20 |
| weich | 0.79 / 0.10 / 0.32 | 0.87 / 0.19 / 0.42 | 0.21 |
| weich + down | 0.79 / 0.04 / 0.33 | 0.88 / 0.13 / 0.44 | 0.25 |
| getrennte Adversaries | 0.72 / 0.03 / 0.24 | 0.84 / 0.06 / 0.40 | 0.19 |
| getrennte Heads | 0.88 / −0.04 / 0.30 | 0.83 / 0.17 / 0.37 | 0.17 |

- **Kernproblem:** Generalisierung. Die Heads lernen die Fehler der U-Net auf deren eigenen
  Trainingsbildern (U-Net-Dice dort 0.92–0.95), auf neuen Bildern derselben Domäne r ≈ 0.
  Unter Shift FN ×5–6 unterschätzt, FP überschätzt (wahr FN ≫ FP).
- Getrennte Adversaries: beide erhöhen vor allem FN (FP-Adv: +0.51 log FP / +0.71 log FN;
  FN-Adv: +0.44 / +0.79) → Hypothese „FN wird zu wenig trainiert“ widerlegt.
- Getrennte Heads: kein Gewinn (nur „zur Vollständigkeit“ trainiert, keine TTA damit gelaufen).

### 2.3 Training: Checkpoint-Auswahl und Lernrate (wichtig für jedes neue Training)

- Checkpoint: `DelayedModelCheckpoint` ([score_adapter.py](../src/model/score_adapter.py)),
  minimaler `val_loss` ab Epoche 150 (`ckpt_start_epoch`, Grund: erst nach Adversary-Pause 40 +
  Warm-up 30 + Beta-Abfall 20 vergleichen). Gleich für alle Heads und den μ̂-Predictor.
- `val_loss` = sauberer Loss + ab Epoche 41 Adversary-Loss → Sprung auf ~2×.
- **LR-Scheduler `ReduceLROnPlateau` (Patience 20) hängt an `val_loss`** und kürzt deshalb viermal ×0.1:
  3.16e-5 → 3.16e-9 (Heads: Kürzungen Ø Epoche ~30…~100; μ̂: ~66…129). **Alle** Head-Checkpoints und
  alle 5 μ̂-Checkpoints (3T→1.5T) enthalten lr = 3.16e-9 → Training ab ~Epoche 60–130 eingefroren.
- Die „besten“ späten `val_loss`-Werte sind Rauschen im Adversary-Anteil (Controller läuft weiter);
  der saubere Anteil am gewählten Checkpoint liegt nur 0.02–0.04 unter dem Mittel.
- Frühere Checkpoints wären nicht besser gewesen (Validierungs-r über Epochenfenster gemittelt flach).
  Achtung: Maximum über einzelne Epochen auf 103 Validierungsbildern ist Auswahlrauschen.
- Neu: `lr_monitor` (`val_loss` | `val_loss_clean`) im FP/FN-Head; für neue Heads `val_loss_clean` nutzen.
- wandb: Projekt `golop-university-of-bonn/MIDL-Extension`; Kurven gecacht in
  `results/fpfn_head/wandb_val_history.csv` und `results/wandb/mu_predictor_threet-to-onepointfivet_val_history.csv`.

### 2.4 TTA mit den Heads (3T→1.5T, label-frei)

| Variante | Dice nach TTA (Start 0.677) |
|---|---|
| μ̂ allein | 0.753 |
| μ̂ + Head hart / weich / weich+down (exp7) | 0.679 / 0.645 / 0.627 |
| nur Head hart / getrennte Adv. (exp8) | 0.651 / 0.637 |

- Der Head meldet weniger FN, die echten FN steigen (1590 → 1690–2110). Die TTA nutzt den Head aus.
- **Mechanismus:** Beim Orakel läuft der Gradient nur über die Maske (Formel), beim Head durch das
  Netz selbst, auch direkt über die adaptierten Features 0/17. Gleicher Wert ≠ gleiche Steigung;
  die Steigungen des Heads sind nie trainiert. „Head-Ausgabe durch richtige Zahl ersetzen“ hilft
  deshalb nicht (Richtung kommt weiter vom Head oder ist null).

### 2.5 Pseudo-Label-Orakel (exp9, 3T→1.5T)

Orakel-Formel mit fester, verschlechterter Karte ĝ statt g (kein Gradient durch ĝ) — die Form, in die
ein dichter Fehlerkarten-Head passen würde (`tta.fpfn_source=pseudo`).

| ĝ | Dice von ĝ | Dice nach TTA |
|---|---|---|
| recall 1.0 (= Orakel) | 1.000 | 0.886 |
| recall 0.75 | 0.925 | 0.843 |
| recall 0.5 | 0.849 | 0.765 |
| recall 0.25 | 0.767 | 0.706 |
| recall 0.0 (ĝ = Maske vor TTA) | 0.677 | 0.677 |
| Label +3 px / −3 px | 0.911 / 0.896 | 0.847 / 0.816 |

- recall r = Anteil der U-Net-Fehler, die in ĝ korrigiert sind (zusammenhängende ~16-px-Flecken).
- Gewinn sinkt gleichmäßig; halb richtiges ĝ schlägt μ̂ allein.
- **Aber die TTA landet immer unter ĝ selbst** (32–73 % von ĝs Gewinn): ein Head mit vollständigem ĝ
  wäre direkt als Segmentierung besser. TTA lohnt nur, wenn sie aus teilweisem ĝ Korrekturen
  überträgt (nicht getestet). TTA-Kurven bei Schritt 100 noch nicht konvergiert; nur ein Seed pro Stufe.

## 3. Offene Punkte (priorisiert)

1. **Dichter Fehlerkarten-Head (vorgemerkt, vom Nutzer gut gefunden, nicht gebaut).** Pro Pixel
   „falsch ja/nein“ (Art folgt aus der Maske), Swivels 0/17/Logits, kleiner Decoder auf 384×384,
   gewichtete BCE + Dice-Loss, 5 Seeds. Entscheidend: realistische Trainingsfehler
   (Shift-Augmentierungen: Kontrast/Gamma, Bias-Field, Rauschen, Unschärfe, Auflösung; optional
   Input-Adversary; ggf. Cross-Fitting), Modellauswahl auf Validierung, Scheduler auf `val_loss_clean`.
   Auswertung: (1) ĝ-Qualität vs. Baselines min(p,1−p) und MC-Dropout (U-Net heißt `dropout-01`,
   ungeprüft), Einordnung in die exp9-Kurve; (2) ĝ direkt als Segmentierung; (3) TTA mit ĝ
   (`fpfn_source=pseudo`, nur Modus „ĝ vom Head“ fehlt). Geschätzt 2–5 h Training pro Seed.
2. Baselines ohne Training für FP/FN-Zahlen: erwartete FP = Σ(1−p) über Maske, FN = Σp außerhalb;
   MC-Dropout. Vergleichsmaßstab für jeden Head.
3. Teil-ĝ-Test (Pseudo-Label nur auf sicheren Pixeln) + mehrere Seeds pro exp9-Stufe.
4. LR-Scheduler-Problem betrifft auch den μ̂-Predictor; ein Neutraining mit `val_loss_clean` wäre
   ein eigener Test (nicht gemacht).
5. Verpasste Prostatas: für FP/FN-Objective außer Reichweite → Erkennung auf Bildebene statt Adaptation.
6. Niedrige Priorität: größerer Dice-Abfall / Controller auf FP/FN-Anstieg (ändert nur die Schrittweite,
   nicht die Richtung), TTA mit den getrennten Heads.

## 4. Code-Stand und Bedienung

- `tta.objective=none` (nur FP/FN-Term) — exp8: [src/apps/exp8_fpfn_only.sh](../src/apps/exp8_fpfn_only.sh),
  Queue [run_fpfn_only_queue.sh](../src/apps/run_fpfn_only_queue.sh) (`SOURCES`, `CKPTS`, `HEAD_TAG`,
  `HEAD_TAG_FN` für getrennte Heads, `SPLITS=threet-to-onepointfivet` setzen).
- Pseudo-Label-Orakel — exp9: [exp9_pseudo_oracle.sh](../src/apps/exp9_pseudo_oracle.sh)
  (`PSEUDO_MODE=recall|morph`, `PSEUDO_LEVEL`), Queue [run_pseudo_oracle_queue.sh](../src/apps/run_pseudo_oracle_queue.sh).
- FP/FN-Head-Optionen: `adv_mode` (joint | split), `count_output` (both | fp | fn), `lr_monitor`,
  `count_target` (hard | soft), `adv_down`. Getrennte Heads: [run_fpfn_sep_heads_queue.sh](../src/apps/run_fpfn_sep_heads_queue.sh)
  (`TAG=fpfnsep` → `fpfnsepfp_k`, `fpfnsepfn_k`); in der TTA kombiniert über `tta.fpfn_head_name_fn`.
- Head-Training-Queue: [run_fpfn_head_queue.sh](../src/apps/run_fpfn_head_queue.sh), Logs `logs/train_<name>_head_<k>.log`.
- Ergebnisordner (alle unter `results/`, nicht versioniert): `latent_tta_fpfnonly_{oracle,head,head_fpfnsplit}`,
  `latent_tta_pseudo_{recall<r>,morph<k>}`, `fpfn_head/`, `wandb/`.
- Notebook neu erzeugen: Generator-Skript lag im Session-Scratchpad (`make_nb_de.py`, nicht im Repo);
  das Notebook selbst ist versioniert und lässt sich mit `jupyter nbconvert --execute` neu rechnen.

## 5. Arbeitsregeln (vom Nutzer)

- GPU-Jobs nur in tmux, nur GPUs 2, 3, 4, 6, 7 (0, 1, 5 gehören anderen; 5 nur nach Rückfrage, wenn frei);
  vor jedem GPU-Start fragen.
- Commits nur auf Nachfrage. Antworten auf Deutsch; Folien/Meeting-Material auf Englisch.
