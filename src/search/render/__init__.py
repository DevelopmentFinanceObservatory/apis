"""Render resolved Specs as confirmation sentences without graph, LLM, or I/O calls.
The renderer is total for well-formed Specs. Import names from their owning submodules.

heading.py builds a Spec's typed heading and byline, tokens.py lays them out as tokens,
and plan.py joins the two for a Spec. how.py, what.py, and window.py resolve axes.
no_data.py and candidates.py render outcomes, and departures.py writes departure text.
envelope.py composes the envelope sentence from the parts, joining definite parts'
headings through join.py. labels.py provides shared helpers.
"""
