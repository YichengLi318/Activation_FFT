# Examples

Two studies, chosen because they test the method from opposite ends.

The first has a known answer. Regulated Chinese verse puts exactly five or seven
characters in every line, so the period is an integer you can write down before
running anything — the experiment can fail and you would notice. The second has
no known answer and needs the intervention half: delete a band, let the model
continue on the damaged signal, sweep depth.

Both find layer 0 uninformative. The poem spectrum is flat there and sharp at
layer 20; deleting a high band there costs nothing. Whatever the token axis
carries, the network builds it rather than receiving it.

## Running these

**Neither runs as shipped.** Each folder has a blank `dataset/`, one script, and
the figures we got. Models and corpora are yours to supply; the numbers below
are a record of what we saw, not something this repository reproduces.

```bash
pip install -e ".[examples]"       # from the repository root
cd examples/01_poem_line_length    # or 02_frequency_locality
```

Drop your data in that folder's `dataset/` and run its script. Both take
`--model` as a Hub id or a local path, print what they measured before they
report anything, and write their figures next to the script. `--help` lists the
rest; each script's docstring gives the dataset format it expects.

Anything under `dataset/` is gitignored, so your corpora stay local.

---

## Why classical Chinese poetry is a good test signal

Finding periodicity in language usually founders on not knowing the period in
advance. Sentence lengths vary, rhythm is statistical, and a peak has nothing to
be checked against.

Regulated verse removes that problem. Written Chinese has no alphabet and no
inflection: one character is one syllable and, in the classical language, nearly
one word. No articles, no plural endings, no tense markers — none of the
grammatical filler that pads an English line. And 近體詩 fixes the count. A
五言 poem has five characters in every line, a 七言 poem seven. Not on average;
the form requires it. Four to eight identical lines follow one another.

So the token sequence is about as close to a square wave as natural text gets:
integer period, small, known beforehand, repeated several times inside a short
window. A method that cannot find period 5 here is broken.

English verse cannot do this. Iambic pentameter fixes ten syllables, but that is
six to ten words and an unpredictable number of subword tokens — and tokens are
the only index the hidden states have. You get a smeared peak with nothing to
compare it to.

The two forms also control each other: same model, same method, two corpora,
peaks that must land in different predictable places. Separating 5 from 7 needs
18 tokens of context, which is four lines either way.

**Check your tokenizer first.** A character is not guaranteed to be a token —
English-heavy tokenizers split Chinese characters into byte pieces, and
Chinese-heavy ones sometimes merge two characters into one. Line-final
punctuation does it more predictably: keep the `，` and a 五言 poem has period
six, not five. `run_poem_spectrum.py` measures tokens per line and prints it
before reporting any peak, and `--separator '，'` shows the shift:

```
separator ''    tokens per line {5: 20} -> look for period 5
separator '，'   tokens per line {6: 20} -> look for period 6
```

---

## 01 — Line length in the spectrum

```bash
python run_poem_spectrum.py --model <model> --poems dataset/poems.json
```

Corpus format is in the script's docstring: one group per line length, each poem
a list of lines without punctuation.

![Layer 0](01_poem_line_length/spectrum_layer_00.png)
![Layer 20](01_poem_line_length/spectrum_layer_20.png)

Seven-character poems, from an earlier run on a model deeper than 32 layers.

Layer 0 is noise — curves wander, the loudest dimension reaches power ~30,
nothing aligns. Layer 20 peaks sharply at f ≈ 0.139 at power ~50,000, with
further peaks near 0.278 and 0.403. Those are f₀, 2f₀, 3f₀: a harmonic stack is
what a repeating non-sinusoidal pattern looks like, and the cheapest evidence
that a peak is not one noisy bin.

The fundamental is period 7.2, which is period 7 on a 72-token grid: a synthetic
period-7 signal in a 72-token window lands on the same bin.

So the periodicity is not in the input representation. It appears with depth.

Two caveats. Only seven-character poems survived — we saw a period-5 peak for
五言 but the figure and the corpus are both gone, so treat that as an
unsupported claim. And the x-axis is frequency starting at 0.0975, because the
plotting code of the time discarded the first seven bins as a workaround for
padding leakage; the tool now bounds the transform instead, and new figures use
a period axis with every bin.

---

## 02 — Which layers depend on which band

```bash
python run_locality.py --model <model> --dataset dataset/your_task.json
```

Dataset is a JSON list of `{"prompt": ..., "answer": ...}`. The script picks
equal-power low and high cutoffs per layer, ablates each, and scores against a
round-trip control.

![AG News](02_frequency_locality/global_task_agnews.png)
![RACE](02_frequency_locality/local_task_race.png)

Accuracy drop against layer, for deleting 20% and 40% of the low band versus the
high band. Read by shape, not height — the two bands hold wildly different
energy, so their heights are not comparable. Comparing a curve against itself
across depth is.

**The bands switch on at different depths.** Deleting the high band costs
nothing at layers 0–4 (0.01 on AG News), peaks at layer 8 (0.45), then decays.
The low band is already at full strength at layer 0 (0.62) and stays flat. One
curve rises, the other does not: dependence on fast-varying structure is built
over the early layers, dependence on slow-varying structure is there from the
input.

**Both switch off together.** Everything collapses between layers 20 and 24
(0.54 → 0.10 on AG News, 0.41 → 0.05 on RACE) and stays near zero to 32. The
last third of the network runs fine with a band of its residual stream deleted.

**Low-band damage saturates, high-band damage does not.** 20% and 40% coincide
for the low band wherever it matters (0.62/0.62, 0.60/0.64, 0.54/0.54). The high
band shows a dose response. The saturation breaks at exactly the layer where the
effect starts dying: at layer 24 on AG News, 20% costs 0.10 and 40% costs 0.26.

One limit on the depth argument: the cutoffs are fixed across layers, but the
energy they enclose is not, because spectral shape changes with depth. The check
is to measure the high band's power share at layers 0, 4 and 8 with
`power_profile` and confirm it does not itself jump. We have not run it.

**Provenance.** The layer axis passes 32, so these runs used a deeper model than
the Qwen3-1.7B the old config assumed, and neither RACE nor AG News is wired
into the current script. The figures predate this code. The task labels are also
arguable — AG News is topic classification that keywords largely solve, RACE
needs a passage integrated — and nothing above depends on which is called local.
