# Portfolio Landing Page Redesign

## What was built

Replaced the original `index.html` with a single-file, static portfolio landing page for Yatin Azad using vanilla HTML, CSS, and JavaScript. The page uses Tailwind via CDN, Google Fonts via `@import`, and GSAP + ScrollTrigger via CDN, with all custom styling and scripting kept inline.

## Changes vs the original site

- Added a full-viewport editorial hero with large stacked Playfair Display typography for `Yatin` / `Azad`.
- Added a sticky navigation bar with scroll-triggered blur, border state, and active section highlighting through `IntersectionObserver`.
- Built a 2-column Selected Work grid with the four requested projects, metric badges, live asset URLs, hover lift, accent borders, and image zoom.
- Reworked research content into a Background section with a large pull quote and two stacked research entries.
- Added a centered credentials line and an auto-scrolling skills ticker.
- Added a large contact/footer section with email, GitHub, LinkedIn, and `Yatin Azad · 2026`.
- Added a custom lagging cursor for fine-pointer devices and disabled it on touch or reduced-motion contexts.
- Added GSAP reveal animations for hero text, sections, and work cards, with `prefers-reduced-motion` fallbacks.
- Added a subtle CSS/SVG grain layer and depth attributes for animated/visual layers.

## Assets

Pulled from the existing repository before editing; Git reported `Already up to date`.

Project images point to the live GitHub Pages asset URLs:

- `https://yatinaz.github.io/assets/tb_preview.png`
- `https://yatinaz.github.io/assets/mandrake_preview.png`
- `https://yatinaz.github.io/assets/fmri_preview.png`
- `https://yatinaz.github.io/assets/eeg_preview.png`

These images are used as complete card previews, so their existing backgrounds are preserved.
