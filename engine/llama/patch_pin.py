"""Patch for llama.dll (Rapidli): a new sequence takes the state cell with its own number (LLAMA_RS_DIRECT=1).

The recurrent memory gives a new sequence the first empty cell. When a sequence starts in a pass of its own (a long
prompt read in bulk while other lanes are idle or not yet started), that is usually not the cell with the sequence's
number; the next pass over all lanes then finds the states out of order and has to gather and copy them (the slow path
the in-place update avoids; with the 8-bit state there is no such path at all). With this patch sequence i starts in
cell i whenever that cell is free, so the states of lanes 0..n-1 always sit in cells 0..n-1.
  python patch_pin.py        (run on the source in src/)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
NL = chr(10)
p = os.path.join(HERE, "src", "llama-memory-recurrent.cpp")
s = open(p, encoding="utf8").read().replace(chr(13) + NL, NL)


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("""    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {""",
    """    // RAPIDLI: cells handed out in this call (a cell stays "empty" until the positions are updated below)
    static const bool pin = getenv("LLAMA_RS_DIRECT") != nullptr && getenv("LLAMA_RS_DIRECT")[0] == '1';
    std::vector<char> taken(pin ? size : 0, 0);

    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {""")

rep("""        if (!has_cell) {
            auto & empty_cell = cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());""",
    """        if (!has_cell) {
            // RAPIDLI: a new sequence takes the cell with its own number when that one is free
            uint32_t target = next_empty_cell;
            if (pin && seq_meta.tail < 0 && cells[seq_id].is_empty() && !taken[seq_id]) {
                target = (uint32_t) seq_id;
            } else if (pin && taken[target]) {
                for (uint32_t j = 0; j < size; ++j) {
                    target += 1;
                    if (target >= size) { target -= size; }
                    if (cells[target].is_empty() && !taken[target]) { break; }
                }
                next_empty_cell = target;
            }
            const bool advance = target == next_empty_cell;
            auto & empty_cell = cells[target];
            GGML_ASSERT(empty_cell.is_empty());""")

rep("""            seq_meta.tail = next_empty_cell;
            // find next empty cell
            if (s + 1 < n_seqs) {
                for (uint32_t j = 0; j < size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= size) { next_empty_cell -= size; }
                    auto & cell = cells[next_empty_cell];
                    if (cell.is_empty()) { break; }
                }
            }""",
    """            seq_meta.tail = target;
            if (pin) { taken[target] = 1; }
            // find next empty cell
            if (s + 1 < n_seqs && advance) {
                for (uint32_t j = 0; j < size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= size) { next_empty_cell -= size; }
                    auto & cell = cells[next_empty_cell];
                    if (cell.is_empty() && !(pin && taken[next_empty_cell])) { break; }
                }
            }""")
if "#include <cstdlib>" not in s:
    rep("""#include "llama-memory-recurrent.h"
""", """#include "llama-memory-recurrent.h"

#include <cstdlib>
""")
open(p, "w", encoding="utf8", newline=NL).write(s)
print("patched: a new sequence takes the state cell with its own number (llama-memory-recurrent)")
