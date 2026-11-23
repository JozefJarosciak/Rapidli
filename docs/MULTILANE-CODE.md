# Code on many lanes

A request for a page or module of many functions (canvas effects, generative art, physics, charts, mini-games, text
utilities) is written by many lanes at once instead of one. Each function is written in several copies side by side,
every copy is checked and run, the best one of each is kept, and the page is put together around them.

## What happens, in order

1. **The list** (one drafted lane, 1-8 s). The model names the functions, one per line, each with a short phrase. With
   a number asked, the lines are numbered and the list is read as it is written: it stops at that number, and when the
   model starts repeating names (asked to invent 64, one pass gives ~32 and then repeats) it goes on from its own lines
   without the repeats. A name that is an earlier one with a number after it (nebula2) is left out first, and used
   only to make up a list that is still short of the number asked.
2. **The copies** (34-66 lanes, 9-12 s). Every function is written 1-5 times on its own lanes, all in one fork from the
   same read of the request. The lanes see only the list (names), not each other's code. Copies that run out of their
   window go on from where they parked, without reading the request again. A copy that took its label for an object
   (`PART.fireworks = function ...`) or wrote the body without its function is put right by its text.
3. **Checks.** Every copy: `node --check`, unknown helper calls, imports where a page cannot have them. For pages, the
   best copies of each function are run in a hidden browser (`smoke.py`): did it throw, did it draw, did it move, was
   it too slow, did it make more drawing calls in one frame than 0.43 per pixel of its canvas (an effect whose work grows
   every frame, or one that paints a full card pixel by pixel: either stalls the whole page).
4. **The page.** Drawing functions go into a gallery made from a template (cards, canvases, one animation loop, each
   function in its own wrapper so one bad function cannot stop the rest; cards off the screen are not drawn); a lane
   writes the title, subtitle and CSS at the same time as the copies. Games get input: click a card to play it with the
   arrow keys, WASD and space; a game that stands still while nobody plays it (no play of its own, or a game over that
   waits for a click) is started again and played by a bot.
5. **Self-test and repair.** The finished page is run; a card that throws or stays blank is swapped for its next copy, and
   every function with no working copy is written again (up to 32, 2-4 copies each, 8 s at most). The best page seen is
   kept.

What the person sees while it works: the function names straight away, then "N lanes wrote their first drafts", "N
copies run in a hidden browser: M draw, move and throw nothing", and "written again: ..." when that happens. The answer
says how many candidates there were and how many were clean.

## Measured (Qwen3.8-35B-A3B IQ3_M, 91-114 lanes, 8k context; through the app, 22 Nov)

| Request | One lane | Many lanes (b12 + b13, four runs) |
|---|---|---|
| 16 canvas effects | 24.8 s, 285 lines, 16/16 animating | 15.8 - 22.5 s, ~490 lines, 16/16 animating |
| 32 canvas effects | 33.8 s, 451 lines, 31/32 drawing | 18.8 - 31.0 s, ~850 lines, 31-32/32 |
| 64 canvas effects | does not fit in 8k | 44 - 46 s, ~1,600 lines, 63-64/64 (b17, final code, three runs) |
| 12 generative art | 36.7 s, 262 lines, 12/12 | 22.6 - 25.9 s, ~450 lines, 12/12 |
| 10 physics | 54.7 s, does not run | 24.4 - 29.3 s, ~430 lines, 7-10/10 drawing, 9 moving (b13) |
| 12 dashboard charts | - | 17.6 - 24.4 s, ~510 lines, 12/12 |
| 12 playable games | - (8 self-playing: 26.8 s, 7/8) | 32.8 - 37.3 s, 9-12/12 drawing |
| 20 text functions page | 13.9 s, 76 lines | 8.4 - 8.9 s, ~225 lines |
| Python module, 20 functions | - | 8.4 s, 307 lines with doctests, 20/20 clean |

Peak while the copies are written: 1,450-2,460 tok/s across 34-66 lanes (one lane: 80-230 tok/s). The many-lane pages
are also two to three times longer for the same request: every function gets its own full attention.

A single program that is one tangle of state (one Tetris game) is still written by one drafted lane: its parts depend
on each other too much to be written apart. It runs in 12-50 s; the page test catches a first draft that hangs or does
not parse and writes it again. (Raced against three ordinary lanes writing the same page, the first working page came
at 27-29 s every time: the lanes share the drafted lane's passes and slow it to 26-30 s. Not used.)

The engine crashed twice in a night of many-lane pages ("CUDA error: an illegal memory access"). The app now notices a
vanished engine within ~6 s and answers again ~26 s after the crash (it was 50-70 s); the answer that was being written
is lost and has to be asked again.

## Demo

Settings: model **Qwen3.8 35B A3B IQ3_M**, Auto length, thinking on or off (not used by this route), web off.
Open the "engine right now" panel to show the lanes filling. Each answer has a **Run** button: the page runs in the side
pane, and Page / Code switches between the result and its source. Close programs that hold the graphics card (a screen recorder
took 3.4 GB: 91 lanes instead of 114-125).

1. **32 canvas effects** (19-31 s, the strongest all-round demo):
   > Write an html/css/js page that shows 32 animated canvas effect functions: starfield, plasma, spiral, fireworks,
   > bouncing balls, rotating wireframe cube, sine waves, particle fountain, matrix rain, Lissajous curves, kaleidoscope,
   > flow field, water ripples, orbiting planets, fractal tree, rainbow tunnel, lava lamp, aurora, snowfall, rain on
   > glass, pulsing grid, radar sweep, DNA helix, galaxy spiral, metaballs, tunnel of squares, electric arcs, bubbles,
   > falling leaves, clock gears, equalizer bars, neon text glow. Each function draws one frame as
   > name(ctx, width, height, time), and each effect runs in its own card with its own canvas, all animating at once, on
   > a dark neon theme.
2. **64 effects, invented by the model** (44-46 s, 66 lanes, 2,250-2,360 tok/s; one lane cannot write this at all):
   > Write an html/css/js page that shows 64 animated canvas effect functions - invent a varied set: space, nature,
   > geometry, particles, retro computer, physics, op-art, weather, music visualizers. Each function draws one frame as
   > name(ctx, width, height, time), and each effect runs in its own card with its own canvas, all animating at once, on
   > a dark neon theme.
3. **12 playable arcade games** (33-37 s, ~60 lanes; every game plays itself, click one to take over):
   > Write an html/css/js arcade page with 12 playable mini-game functions: paddle tennis, snake, brick breaker, alien
   > invaders, asteroid shooter, flappy bird, falling blocks, maze chase, road crossing frog, missile defense, light
   > cycles, endless runner. Each game is a function that draws one frame as name(ctx, width, height, time, input), in
   > its own card, retro neon arcade style.
4. **Live analytics dashboard** (18-24 s):
   > Write an html/css/js analytics dashboard page with 12 chart functions that each draw animated sample data: bar
   > chart, line chart, pie chart, donut, stacked area, scatter plot, radar chart, heatmap, gauge, sparklines, histogram,
   > candlestick chart. Each function draws one frame as name(ctx, width, height, time) in its own card, clean dark
   > dashboard style.
5. **Generative art gallery** (14-26 s):
   > Write an html/css/js generative art gallery page with 12 generative art functions: Mondrian grid, Truchet tiles,
   > flow lines, circle packing, Voronoi cells, spirograph, Perlin landscape, recursive squares, dot halftone, string
   > art, crystal snowflake, aurora waves. Each function draws one frame as name(ctx, width, height, time), slowly
   > animating, each in its own framed card, museum style.
6. **Physics playground** (24-29 s; one lane: 55 s and broken):
   > Write an html/css/js page with 10 physics simulation functions, each animating in its own canvas card: pendulum,
   > double pendulum, spring mass, bouncing balls with gravity, boids flocking, cloth, orbital gravity, wave on a string,
   > falling sand, Newton's cradle. Each function draws one frame as name(ctx, width, height, time).
7. **Text tools page** (9 s, the quickest):
   > Write html/css/js page, that features demo of 20 functions for working with text: slugify, palindromes, word
   > counts, title case, and similar.
8. **Not a page: a Python module** (8 s, 300 lines with doctests):
   > Write a Python utility module with 20 functions for working with text: slugify, palindromes, word counts, title
   > case, and similar.

What makes a request go to the many-lane writer: a number (or "several", "many", "a set of", ...) before "functions"
(or helpers, utilities, methods, algorithms, snippets), and a sign that code is wanted (html/css/js, javascript, python,
code, module, ...). "Canvas", "draws", "animated", "art", "simulation" and the like make it a gallery of canvases;
"playable", "game" or "arcade" makes it a gallery of games. A question about functions is not a request for them: "What
are 5 functions of the liver?" or "What are the 4 functions of a compiler?" stay text answers.

## Settings

| Variable | Default | What it does |
|---|---|---|
| `CODE_LANES_MAX` | 40 (62 for games) | lanes for the copies (64 functions take one lane each) |
| `CODE_GO_ON` | 2 (3 for games) | copies of one function that may go on when none finished |
| `CODE_LANES_BUDGET` | 25 s (35 for games) | the copies stop here |
| `CODE_SECOND_BUDGET` | 8 s | functions written again |
| `CODE_RUN_COPIES` | 1 | run the copies in a hidden browser before choosing |
| `RAPIDLI_VAULT` | 80 | parked states the engine keeps (copies park in them) |

Measured and not used: a time budget per frame (no difference), two copies each for 32 functions (31-32/32 either
way, 32-34 s instead of 25-28), three lanes racing the drafted lane on one-program pages (see above).

Files: `app/codelanes.py` (all of the above), `app/smoke.py` (the hidden browser test), `app/codecheck.py` (checks),
`app/pagelanes.py` (one-lane page repair). A page of 64 canvases needs 11-18 s in the headless browser, so a page's
check is allowed 40 s.
