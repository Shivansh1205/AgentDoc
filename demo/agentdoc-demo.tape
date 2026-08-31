# VHS tape script for the AgentDoc demo GIF.
# Docs: https://github.com/charmbracelet/vhs
#
# Run from the demo/ directory: `vhs agentdoc-demo.tape`
# Produces agentdoc-demo.gif in this directory.
#
# Before recording for real: set GROQ_API_KEY in your shell (or a .env file
# in this directory) so the `diagnose` calls actually classify instead of
# erroring. The commands below assume it's already set, matching the
# "install -> set key -> diagnose clean -> diagnose flawed" flow.

Output agentdoc-demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1100
Set Height 650
Set Theme "Dracula"
Set TypingSpeed 40ms

Type "pip install agentdoc"
Sleep 500ms
Enter
Sleep 2s

Type "export GROQ_API_KEY=gsk_your_key_here"
Sleep 500ms
Enter
Sleep 1s

Type "clear"
Enter
Sleep 500ms

Type "# A clean run - no issues"
Sleep 500ms
Enter
Type "agentdoc diagnose examples/langgraph_trace_example.json"
Sleep 500ms
Enter
Sleep 4s

Type "clear"
Enter
Sleep 500ms

Type "# A trace with real problems injected"
Sleep 500ms
Enter
Type "agentdoc diagnose examples/langgraph_trace_flawed_example.json"
Sleep 500ms
Enter
Sleep 6s
