# Data decision: the P&L series

Decided: 2 September 2026, on the evidence in `outputs/roll_audit`.

## The problem

The regime signal is built on the research project's frozen `CL=F` series
(daily close, 4 January 2010 to 30 December 2024, hash in the research
`DATA_MANIFEST.md`). That series is fine for detecting regimes. It is not a
clean tradable return. Yahoo splices front-month contracts with no
adjustment, so on the day the front contract changes the recorded price
change mixes a market move with the calendar spread between the two
contracts. A position holder does not earn that spread.

## What the audit found

`01_roll_audit.py` reconstructs the NYMEX CL expiry date for each of the 180
delivery months in the sample (three trading days before the 25th of the
prior month, using the file's own dates as the calendar) and flags the expiry
day and the day after it. With 20 and 21 April 2020 removed (the May 2020
contract expired on 21 April, so those two days sit inside the window):

| Days | Count | Mean abs. change | Median abs. change | Std. dev. | Sum |
|---|---:|---:|---:|---:|---:|
| Expiry day | 179 | $1.01 | $0.85 | $1.34 | +$11.34 |
| Day after expiry | 180 | $1.25 | $1.02 | $1.60 | -$0.25 |
| All other days | 3,409 | $1.11 | $0.84 | $1.57 | -$13.35 |

Three things follow.

1. The day after expiry is the roll day. Its moves are about 12% larger by
   mean and 20% larger by median than ordinary days. The expiry day itself
   looks like any other day. So Yahoo switches contracts after expiry, and
   the extra size on the following day is the calendar spread.
2. The contamination is small. A typical spread adds a few tens of cents to
   a day whose scale is a dollar. It touches 180 of 3,770 days (4.8%).
3. There is no material drift. The flagged days sum to about +$11 over
   fifteen years, or six cents a roll. Contango and backwardation have
   roughly cancelled over the sample. A long position on the spliced
   series is not systematically flattered or penalised.

## Decision

- The P&L series is the close-to-close change on the research frozen file.
  No new download. No back-adjusted series is available free, and the
  audit shows the error is small.
- The day after each expiry is the designated roll day. Every backtest runs
  with three treatments of those days and the gate result must hold under
  all three: (a) raw changes; (b) roll-day change set to zero; (c) roll-day
  rows dropped from the P&L statistics.
- The regime signal is always computed on the raw series, under every
  treatment. The model was selected on that series and is not re-estimated
  for the trading project.
- Recorded limitation: the P&L on 4.8% of days carries a spread component
  of roughly $0.2 to $0.4. Because the test is a difference between two
  rules facing the same series, the component cancels unless the two rules
  hold systematically different positions on roll days. That is checked
  by treatment (c).

## Later improvement, not now

A roll-free daily series exists free from the EIA (Cushing WTI spot, series
RWTC). Spot is not tradable and has its own quirks, but it would serve as a
robustness series if a result ever turns on the roll treatment.
