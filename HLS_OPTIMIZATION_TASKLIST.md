# My YOLOv2 INT16 HLS IP Optimization Task List (aim: 10×+)

Right now I’m at roughly **0.1 FPS** end-to-end. My near-term goal is **≥1 FPS (10×)**, and the long-term goal is a **real-time(-ish)** camera demo on KV260 someday. I also want to keep detection performance intact (no “speed at all costs” hacks that break accuracy).

This is my step-by-step checklist for speeding up the `YOLO2_FPGA` HLS IP. I’ll work through it top-to-bottom and tick items off as I go. It’s ordered from **highest leverage / lowest risk** → **larger changes / higher risk**.

**Where I’m starting from (current repo state):**
- My INT16 HLS script targets `clk_period 5.0` (200 MHz) in `vitis/yolo2_int16_cli.tcl`
- HLS reports show 32-bit AXI on `DATA_BUS1` (`m_axi_DATA_BUS1_RDATA` width 32) in `yolo2_int16/solution1/syn/report/YOLO2_FPGA_csynth.rpt`
- My convolution compute loop is explicitly `PIPELINE II=3` in `hls/core/core_compute.cpp`

---

## 0) Baseline + measurement hygiene (I do this first)

- [x] I record current “known-good” KV260 throughput (ms/frame and FPS) for a fixed workload (same video/image, same thresholds, same infer-every).
- [x] I capture HLS summary metrics from `yolo2_int16/solution1/syn/report/YOLO2_FPGA_csynth.rpt`:
  - Target clock (ns), estimated clock (ns)
  - DSP / LUT / FF / BRAM totals
- [x] I capture Vivado **post-route** timing at 200 MHz:
  - WNS/TNS in `report_timing_summary`
  - Optional: `report_power` (even with default activity)
- [x] I keep a small “perf log” table (markdown/CSV) so every change has: `{change, HLS fmax, HLS II, Vivado WNS@freq, board ms/frame}`.

---

## 1) Reports I can generate automatically (HLS + Vivado + power)

Before I start changing the accelerator, I want a repeatable way to measure progress and catch regressions quickly.

- [x] I add a small parser for HLS XML reports:
  - Inputs: `yolo2_int16/solution1/syn/report/*_csynth.xml`
  - Output: one human-readable summary table (markdown/CSV).
- [x] I add a Vivado batch flow to dump:
  - `report_utilization`
  - `report_timing_summary`
  - `report_power`
- [x] I define a single “report bundle” directory layout so comparisons are easy:
  - Example: `reports/<date>/<variant>/hls.csv`, `vivado_timing.rpt`, `vivado_util.rpt`, `vivado_power.rpt`

I’ll call this done when:
- [x] I can run one command per variant and get a consistent, comparable report bundle.

---

## 2) Fix DDR/AXI bandwidth first (often the biggest limiter)

My goal here is to make sure the IP can actually pull/push data fast enough before I crank up MAC parallelism.

- [ ] I confirm AXI port widths in the exported RTL (I want ≥128b, ideally 256b+):
  - Check HLS report port widths (e.g., `m_axi_*_RDATA/WDATA` bitwidths).
- [ ] I increase AXI port widening (try 256 or 512) on all heavy ports:
  - `Input` (`DATA_BUS_IN`), `Output` (`DATA_BUS_OUT`), `Weight` (`DATA_BUS1`)
  - Implement via `max_widen_bitwidth` on `m_axi` interfaces (pragma or `set_directive_interface`).
- [ ] I increase outstanding transactions:
  - Right now I’m using `num_read_outstanding=1` / `num_write_outstanding=1` in `hls/models/yolov2/yolo2_accel.cpp`
  - I’ll try values like 8/16/32 and re-synthesize; I’ll watch for diminishing returns.
- [ ] I increase burst lengths (if the interconnect + DDR controller can accept it):
  - Start by raising `max_read_burst_length` / `max_write_burst_length` gradually.
- [ ] I make sure `DATA_BUS_IN`, `DATA_BUS_OUT`, `DATA_BUS1` map to **separate HP ports** in Vivado (avoid sharing one DDR port).
- [ ] If widening doesn’t “stick” with `int16_t*` ports, I introduce packed ports:
  - Example direction: use `ap_uint<256>*` for DDR + unpack into on-chip `int16` arrays.
  - (This is more invasive, but it’s the cleanest way to guarantee wide bursts.)

I’ll call this done when:
- AXI ports are wide (≥128b) and HLS + Vivado show stable burst behavior, and KV260 runtime improves (or at least doesn’t regress).

---

## 3) Make convolution throughput faster (II + parallelism)

My conv kernel is the long pole; I need to fix it before almost anything else matters.

- [ ] I change the conv loop structure to target **II=1** on the core MAC pipeline.
  - Today `PIPELINE II=3` is set at the `tc` loop level in `hls/core/core_compute.cpp`.
  - I’ll consider moving the pipeline deeper and/or restructuring loops so the pipelined loop body is smaller.
- [ ] I fully unroll `tn` (Tn=4) inside the MAC:
  - This should cost ~4× DSP on that dimension but can cut latency.
- [ ] I partially unroll `tm` (start with factor 2, then 4, etc.):
  - I’ll watch LUT growth (I’m already LUT-heavy).
- [ ] I make sure weight and output arrays remain compatible with unrolling:
  - Keep `weight_buffer` and `output_buffer` partitioning aligned with your unroll factors.
- [ ] I remove/avoid loop-carried dependencies that block II=1:
  - Keep the accumulator update pattern simple for HLS to schedule.

I’ll call this done when:
- HLS reports show the critical loop at II≈1 and DSP usage rises (I have headroom), and runtime improves.

---

## 4) Reduce LUT pressure (helps both II and max frequency)

My current design uses a lot of LUTs relative to DSP. Reducing LUT-heavy math should help both timing (Fmax) and II/unrolling.

- [ ] I replace `int64_t` math in the hot loop with bounded-width arithmetic:
  - Use `ap_int<W>` / `ap_fixed<...>` for partial sums and scaling.
  - Choose widths based on worst-case bounds (document the bound).
- [ ] I hoist constant computations out of pipelined loops:
  - Shifts, rounding constants, clamp parameters.
- [ ] I use saturating helpers that synthesize compactly (avoid repeated compares inside the innermost loops).

I’ll call this done when:
- LUT drops meaningfully, and either HLS estimated clock improves or I can unroll more without timing collapse.

---

## 5) Tune tiling parameters (`Tn/Tm/Tr/Tc`) systematically

Tiling changes can deliver big gains by reducing loop overhead and increasing reuse, but I want to do it systematically (and not randomly).

- [ ] I create a controlled sweep plan (change one parameter at a time):
  - `Tn` (input channel parallelism)
  - `Tm` (output channel parallelism) — also reduces number of `m` tiles
  - `Tr/Tc` (spatial tile sizes)
- [ ] I automate regenerating `hls/core/params.hpp` (currently hardcoded in `scripts/hw_params_gen.py`).
- [ ] After each sweep point:
  - Re-run HLS synth, record utilization + II
  - Re-run Vivado impl at target frequency (or at least check if it’s still routable)

I’ll call this done when:
- I find a “sweet spot” where throughput improves with tolerable BRAM/LUT/DSP and timing is still closable.

---

## 6) Make ping-pong actually overlap (DATAFLOW / streaming)

My code is structured like ping-pong, but HLS can still schedule big chunks serially unless I force overlap (DATAFLOW / streams).

- [ ] I add `#pragma HLS DATAFLOW` at the right level (I’ll start in smaller sub-functions first).
- [ ] I convert load/compute/store boundaries to `hls::stream` where practical.
- [ ] I replace `memcpy`-based staging with explicit burst-read loops that feed streams.
- [ ] I use the HLS schedule viewer to confirm load/compute/write overlap.

I’ll call this done when:
- The schedule shows concurrent phases (DDR read overlaps compute, compute overlaps DDR write), and runtime improves.

---

## 7) Clock scaling plan (200 → 250 → 300 MHz)

Raising clock alone won’t give big gains if I’m bandwidth/II-limited, but it’s still worth doing once the design is “clean”.

- [ ] I sweep HLS `clk_period` in `vitis/yolo2_int16_cli.tcl`:
  - 5.0 ns (200 MHz) → 4.0 ns (250 MHz) → 3.333 ns (300 MHz)
- [ ] For each point, I build in Vivado and verify **post-route**:
  - Accept only if WNS ≥ 0 (ideally with margin)
- [ ] If 300 MHz fails timing:
  - Reduce unroll factors, reduce bitwidth/LUT pressure, add internal pipelining, and retry.

I’ll call this done when:
- I have a stable set of (frequency, utilization, throughput) points and I can justify the chosen clock.

---

## 8) Stretch goals (bigger refactors, bigger potential gains)

- [ ] I separate conv/pool/reorg into dedicated IPs (or at least separate top functions) to let HLS optimize each path harder.
- [ ] I add a second compute engine (spatial or channel split) and run them in parallel (requires DDR bandwidth + careful arbitration).
- [ ] I reduce DDR round-trips between layers (true streaming between layers) — highest effort, highest payoff.
