---
name: Grille Lab
description: A quiet, legible local recording bench.
colors:
  ink: "#193b36"
  muted: "#52665e"
  accent: "#14664e"
  page: "#f5f6f0"
  line: "#d3ddd2"
  stage: "#e5ece1"
  record: "#145f49"
  record-muted: "#d1ebdf"
  white: "#fff"
  timeline: "#c7d5c5"
  button-border: "#abbcaf"
  field-border: "#b6c5b8"
  focus: "#c38028"
  error: "#a22c24"
  progress-track: "#d8e1d4"
typography:
  display:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif'
    fontSize: "clamp(30px, 3.4vw, 44px)"
    fontWeight: 600
    lineHeight: 1.13
    letterSpacing: "-.035em"
  cue:
    fontSize: "27px"
    fontWeight: 600
    letterSpacing: "-.025em"
  title:
    fontSize: "18px"
    fontWeight: 600
  body:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif'
    fontSize: "15px"
    lineHeight: 1.65
  label:
    fontSize: "13px"
  detail:
    fontSize: "12px"
rounded:
  field: "6px"
  button: "7px"
  stage: "14px"
spacing:
  control-gap: "9px"
  field-gap: "18px"
  section: "24px"
  inset: "28px"
  columns: "64px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.white}"
    rounded: "{rounded.button}"
    padding: "11px 15px"
    typography: "{typography.label}"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.button}"
    padding: "11px 15px"
  field:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.field}"
    padding: "10px"
    width: "100%"
  recording-stage:
    backgroundColor: "{colors.stage}"
    textColor: "{colors.ink}"
    rounded: "{rounded.stage}"
    padding: "22px 28px 24px"
  recording-stage-active:
    backgroundColor: "{colors.record}"
    textColor: "{colors.white}"
    rounded: "{rounded.stage}"
---

# Design System: Grille Lab

## Overview

**Creative North Star: "Local recording bench"**

This document captures the implemented interface in `style.css` and `index.html`. A pale sage canvas, compact labeled controls, and a large directional cue keep attention on the current recording action. The working surface is calm and practical for personal laptop use.

**Key Characteristics:**
- Sage surfaces and forest green recording emphasis.
- System typography and compact, labeled controls.
- Flat separation and clear arrow geometry.

## Colors

The frontmatter records the implemented palette. Accent green identifies primary actions and progress; deeper recording green fills the active cue. Ink and muted green distinguish primary and supporting text. Sage page and stage surfaces, white fields, and pale lines define regions. Amber identifies keyboard focus; red identifies notice errors. The active cue uses white text and a pale green status line.

## Typography

Use the system sans-serif stack throughout ordinary interface text. The display role introduces the task; the cue role identifies the current movement. Titles identify sections, body text explains the next action, and smaller labels carry setup and status. Introductory copy is limited to (62ch); cue instructions to (46ch). Microphone level figures use tabular numerals. Technical settings use the browser monospace face at (10px).

## Layout

The current desktop collector uses a centered container with a maximum width of (1240px) and horizontal padding of (40px). The flexible recording column sits beside a (280px) setup column, separated by the column gap token and a fine vertical divider. Layout padding is (44px) vertically; the sidebar inset uses the inset token.

At widths up to (900px), horizontal padding becomes (24px), the sidebar becomes (240px), and the column gap becomes (28px). At widths up to (680px), the layout stacks, page padding becomes (20px), the sidebar divider moves to its top, and action buttons grow to share available width. The cue text becomes (24px). These describe the present collector composition, not a required layout for every future screen.

## Elevation & Depth

The interface uses no shadows. Tonal surfaces, whitespace, and thin borders establish depth. Recording emphasis comes from the stage's green background and light text.

## Shapes

Fields and buttons use compact rounded corners; the larger cue surface has the broadest radius. Dividers and progress tracks are straight. The directional icon is an unfilled SVG with rounded caps and joins, stroke width (5), and a (100px) rendered height. Its clear vertical stem and arrowhead remain legible at a glance.

## Components

- **Buttons:** primary buttons use accent green; secondary buttons use transparent fill and a muted green border. Minimum height is (42px). Hover applies `brightness(.91)`; disabled buttons use opacity (.45) and a not-allowed cursor.
- **Fields:** white selects and textarea use the field radius, a fine field-border stroke, and visible labels above. The textarea resizes vertically. The confirmation checkbox uses accent green.
- **Focus:** buttons, links, summary, selects, textarea, and inputs share an amber outline of (3px), offset by (3px).
- **Recording stage:** the current cue is centered in a sage surface with minimum height (328px), reduced to (320px) on narrow screens. The active recording state changes the entire surface to recording green. A thin (4px) timeline communicates trial progress alongside the text cue.
- **Header:** a bold (20px) brand and muted (14px) descriptor balance a compact connection status. The descriptor disappears at the first breakpoint.
- **Progress and history:** a (6px) accent progress bar supports explicit saved counts; recent recordings use compact rows separated by thin lines.

## Do's and Don'ts

- **Do** keep action labels and recording status explicit alongside color changes.
- **Do** preserve the clear vertical arrow geometry and strong active-cue contrast.
- **Do** retain visible keyboard focus and labeled setup fields.
- **Don't** replace the incumbent flat surfaces with decorative shadows or gradients when extending this visual system.
- **Don't** rely on color alone to convey recording progress or errors.
