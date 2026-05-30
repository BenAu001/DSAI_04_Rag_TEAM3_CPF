# AGENT TASK: Convert Current RAG UI into Left-Right Agentic RAG Layout

## Goal

Redesign the current HTML RAG interface from a centered vertical layout into a left-right dashboard layout similar to the target reference image.

The new layout should have:

- A fixed left sidebar for Corpus and Tools
- A main conversation area on the right
- A top header bar
- A bottom question input area
- A cleaner agent-style RAG interface

---

## Current UI Problem

The current screen is vertically stacked:

- Title at top
- Status bar
- Question box
- Answer box
- Sources listed under answer

This works, but it does not look like an agentic RAG workspace.

---

## Target Layout

Create a two-column layout:

```text
 ------------------------------------------------------
| Header: RAG Corpus / Agentic RAG / New Conversation |
 ------------------------------------------------------
| Left Sidebar        | Main Chat / Answer Area        |
|                     |                                |
| Corpus              | Intro / Agent responses        |
| - PDF 1             |                                |
| - CSV 1             |                                |
|                     |                                |
| Tools               |                                |
| - search_pdf()      |                                |
| - search_csv()      |                                |
| - web_search()      |                                |
|                     |                                |
 ------------------------------------------------------
|                     | Suggested Questions            |
|                     | Input box + Ask button          |
 ------------------------------------------------------