# Recording the AgentDoc demo GIF

Goal: a short (~20-30s) terminal recording showing install -> set API key ->
diagnose a clean trace -> diagnose a flawed trace -> see the diagnosis. This
doc gives you the exact commands, in order, plus two recording options.

## Recommended: VHS (scripted, reproducible)

[VHS](https://github.com/charmbracelet/vhs) (free, open source, from
Charm) records a terminal session from a text script, so the recording is
byte-for-byte reproducible if you need to re-record after a UI tweak — no
re-typing, no timing mistakes to redo.

**Install** (pick one):

```bash
# via Go
go install github.com/charmbracelet/vhs@latest

# via Homebrew (macOS/Linux)
brew install vhs

# via winget (Windows)
winget install charmbracelet.vhs
```

VHS also needs [`ttyd`](https://github.com/tsl0922/ttyd) and
[`ffmpeg`](https://ffmpeg.org/) on PATH — `brew install ttyd ffmpeg` covers
both on macOS; see VHS's README for other platforms.

**Record:**

```bash
cd demo
vhs agentdoc-demo.tape
```

This produces `demo/agentdoc-demo.gif`. Preview it, and if it looks good,
add it to the README where the placeholder comment is (see the bottom of
this file).

**If you tweak the script**, just re-run `vhs agentdoc-demo.tape` — it's
fully deterministic.

## Alternative: asciinema + agg (live recording)

If you'd rather record live instead of scripting it, use
[asciinema](https://asciinema.org/) to record and
[agg](https://github.com/asciinema/agg) to convert to GIF:

```bash
pip install asciinema   # or: brew install asciinema

asciinema rec agentdoc-demo.cast
# ... run the commands below, then Ctrl+D to stop recording ...

agg agentdoc-demo.cast agentdoc-demo.gif
```

Run these commands, in order, once recording starts (pause ~1-2s between
each so the GIF doesn't feel rushed):

```bash
pip install agentdoc

export GROQ_API_KEY=gsk_...          # use your own key; get one free at console.groq.com/keys

agentdoc diagnose examples/langgraph_trace_example.json

agentdoc diagnose examples/langgraph_trace_flawed_example.json
```

Tips for a clean take:
- Use a terminal window sized around 100x30 - wide enough that panels don't
  wrap awkwardly, short enough to fit a GIF without excessive scrolling.
- Clear the terminal (`clear`) right before starting, and right before each
  command if you want a cleaner cut.
- The flawed-trace output is fairly long (5 flagged failures across all 3
  categories) - if the full output feels like too much for a short GIF,
  scroll/crop to show just the summary panel plus 1-2 example failure
  panels, since that's enough to convey the effect.
- Classification is LLM-based, so wording/confidence will vary slightly
  between takes - that's expected and fine.

## Once you have the GIF

1. Save it as `demo/agentdoc-demo.gif` (keep it in the repo so the README
   link works, or host it elsewhere and just link out — either works).
2. In `README.md`, replace the HTML comment placeholder right after the
   intro paragraph with:

   ```markdown
   ![AgentDoc demo](demo/agentdoc-demo.gif)
   ```
