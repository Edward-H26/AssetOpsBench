# Episode scenario2007_seed000000

Config mentor_description_scn1, agent causal (granite-3.3-8b-instruct), servers iot, fmsr, wo, task, seed 0, clock mode declared, guidance level 2.
Task text: Asset hyd_1 (HYDRAULIC_PUMP, Hydraulic pump condition-monitoring rig, at site MAIN), with one child asset per component: hyd_1_cooler, hyd_1_valve, hyd_1_pump and hyd_1_accumulator. The performance monitor flagged the unit. For asset hp_1, using the last 100 days up to and including 2027-01-30, can sensor data reveal the onset of severe internal pump leakage? Which sensor anomalies support this? Answer through the tools: read the telemetry of hyd_1 with the iot server, submit exactly one diagnosis, or one declaration of insufficient evidence, with the task server, create the work orders that restore the unit with generate_work_order of the wo server, and finish with a summary. The diagnosis states the condition level of each component at the cycle it is submitted, and it is scored against the hidden condition at that cycle. Every reading is a recorded cycle, and while the condition levels stay the same the unit replays the same 5 cycles in a loop, so its readings repeat every 5 cycles it runs under unchanged levels, and the 5 cycles recorded before the episode started are other recordings outside that loop. Every answer lists the notifications of the public events since the previous call. The fix history, list_workorders of the wo server, lists the work orders closed on this rig before the episode. An emergency work order arrives sooner at a surcharge but holds the component at its healthy level only for a limited number of cycles, after which the component returns to its earlier level unless a standard order on it closes first. A repair takes only when every component it depends on is at its healthy level at its close. The work orders created one after another, with no other call between them, form one technician visit with one visit charge that repairs them in priority order.

Condition levels: cooler: 3, 20, 100 (healthy 100); valve: 73, 80, 90, 100 (healthy 100); pump: 0, 1, 2 (healthy 0); accumulator: 90, 100, 115, 130 (healthy 130).

Telemetry. A row of hyd_1 holds one recorded cycle: its timestamp and, for every channel, the mean under the installed sensor name and the standard deviation under that name with _std appended. Name the sensors you need in history, since a history without a sensor list returns every field of every row. Row timestamps carry no time zone and history bounds take the same form, while work order dates carry the offset +00:00. A history window that reaches cycles not yet recorded first lets the rig run them, 1 cycle each, and returns and charges them.
history and latest_reading charge every cycle they return at the price per cycle of the cheapest sensor group whose fields cover the fields returned, unless a group paid earlier for that cycle covers them, so reading a paid row again is free:
  - tier1 costs 1 per cycle and returns TS1_Temperature_C_1Hz, TS2_Temperature_C_1Hz, TS3_Temperature_C_1Hz, TS4_Temperature_C_1Hz, VS1_Vibration_mm_per_s_1Hz, CE_CoolingEfficiency_percent_1Hz
  - tier2 costs 3 per cycle and returns TS1_Temperature_C_1Hz, TS2_Temperature_C_1Hz, TS3_Temperature_C_1Hz, TS4_Temperature_C_1Hz, VS1_Vibration_mm_per_s_1Hz, CE_CoolingEfficiency_percent_1Hz, FS1_VolumeFlow_l_per_min_10Hz, FS2_VolumeFlow_l_per_min_10Hz, CP_CoolingPower_kW_1Hz
  - tier3 costs 10 per cycle and returns PS1_Pressure_bar_100Hz, PS2_Pressure_bar_100Hz, PS3_Pressure_bar_100Hz, PS4_Pressure_bar_100Hz, PS5_Pressure_bar_100Hz, PS6_Pressure_bar_100Hz, P1_MotorPower_W_100Hz, FS1_VolumeFlow_l_per_min_10Hz, FS2_VolumeFlow_l_per_min_10Hz, TS1_Temperature_C_1Hz, TS2_Temperature_C_1Hz, TS3_Temperature_C_1Hz, TS4_Temperature_C_1Hz, VS1_Vibration_mm_per_s_1Hz, CE_CoolingEfficiency_percent_1Hz, CP_CoolingPower_kW_1Hz, SE_EfficiencyFactor_percent_1Hz

Tools by server (class, duration in cycles, price):
- iot, with site_name MAIN and asset_id hyd_1: sites, asset_ids, assets, asset_detail, installed_sensors, measured_sensors, find_assets_by_sensors, stream_extent and sensor_coverage (read, 0, free) describe the registry and the recorded stream. history(site_name, asset_id, start, end, sensors, limit, cursor) (read, the cycles it plays, price per cycle by group) returns the recorded rows of a window, and latest_reading(site_name, asset_id, sensor) (read, 0, price per cycle by group) returns the newest recorded row.
- fmsr: get_failure_modes(asset_class) (read, 0, free) returns the failure mode codes of a component class, cooler, valve, pump or accumulator, with their source.
- wo, with site_id MAIN: list_workorders, get_workorder, get_workorder_tasks, get_workorder_costs, get_workorder_actuals_vs_planned, get_workorder_kpis, get_schedule_calendar, get_my_assigned_workorders and get_failure_codes (read, 0, free) read the work orders of this episode and the failure code catalog. generate_work_order(description, asset_num, site_id, priority, work_type, reported_by, location, notes, wonum, aob_source) (write, 0, part price plus the repair charge of 20, plus a visit charge of 15 for the order that opens a visit) asks a technician to restore the component of a child asset to its healthy level, with work_type CM and an empty wonum, since the environment numbers the orders wo-1 and up, and aob_source an object with the prediction_id and the evidence_ids behind the order. With work_type EM the order is an emergency order, which forms a visit of its own, arrives 1 cycle after its creation at 2.0 times the part and repair price, and holds the component at its healthy level for 10 cycles after its close. A component that an emergency order still holds when the episode ends is scored at the level it had before that order. Parts: cooler 20, valve 20, pump 30, accumulator 20. The technician arrives after the lead time and repairs while the rig stands still, then the order closes. Lead times: cooler 3, valve 3, pump 3, accumulator 3. Repair downtimes: cooler 1, valve 1, pump 2, accumulator 1. A work order is refused before a diagnosis or a declaration, and while an order on the same component is open. An order on a healthy component is executed and counted as unnecessary.
- task: get_asset_profile() (read, 0, free) returns these rules, the child assets, the normal ranges of the channel means under the record fields and the tool table.
  submit_diagnosis(cooler, valve, pump, accumulator, evidence_ids) (write, 0, free) records one condition level per component, accepted once and never after a declaration. A wrong diagnosis costs the misdiagnosis price of 15, the call-out that it wastes. The price is added to the cost of the episode when the episode is scored and does not fall from the budget. declare_insufficient_evidence(sensor_group, cycles, evidence_ids) (write, 0, free) records that the evidence cannot separate the remaining candidates and names the reading that would, accepted once and never after a diagnosis, and work orders stay allowed afterward. It must cite at least one reading already read, and a declaration that cites none is scored as a failure. finish(summary) (write, 0, free) ends the episode. With no further call the rig then runs until the open work orders close while the downtime accrues, and the episode is scored after the last close.
evidence_ids of the task tools is a comma separated list of identifiers, possibly empty, and aob_source.evidence_ids a list of them: obs/<timestamp>/<field> for a value of a telemetry row, with the timestamp and the field as the row shows them, wo/<wonum> for a work order, fmea/<source>/<code> for a failure mode and alarm/<cycle>/<channel> for the alarm of a notification. aob_source.prediction_id is empty in this episode.
Every answer ends with one JSON line with the price of the call, the budget left, the steps left, the cycle and the notifications since the previous call of the work orders executed and closed, the alarms of the sensor groups already read and the failures, then the end of the episode. An alarm is raised once per channel in an episode, so a channel that leaves its normal range again after a repair raises no second alarm, and only a new reading of the telemetry shows it.

Downtime accrues per cycle while a component is below its healthy level (cooler: level 100 0, level 20 1, level 3 2; valve: level 100 0, level 90 1, level 80 2, level 73 3; pump: level 0 0, level 1 1, level 2 2; accumulator: level 130 0, level 115 1, level 100 2, level 90 3) and counts toward the cost of the episode, while the budget limits the prices alone. Budget: 300. Horizon: 60 cycles. Step cap: 60 tool calls, refused ones included. A turn that calls no tool of the rig still takes one step, and as many requests in a row without a call as the step cap allows end the episode. Time advances only inside tool calls, by their declared durations. The episode ends on finish, at the horizon, at the step cap, after 1800 seconds of wall time, or when an action would exceed the budget; a component at its worst level for 10 cycles is a failure that adds 50 to the cost, and only a repair on it that has closed by then averts it. If the wall limit ends the episode while every call is answered or while your reply is being written, the episode ends there and is scored. A refused call costs nothing but counts as a step, and a call with a missing argument or an argument of the wrong type is refused.

## Anchored scenario assumptions

- A1 (mapping): the asset hp_1 of the scenarios is the hyd_1 file of the AssetOpsBench repository, which lays the recorded cycles out as one per day from 2022-07-01, so the recording cycle of a date is its number of days since 2022-07-01 (the cycle_index of the facts file; the hyd_1 file numbers the same cycles from 1).
- A2 (dates): the dates were chosen for the faults the scenario texts name, and the facts file does not show that fault on some of those days, so the hidden condition of an anchored episode is the recorded condition of the day and not the fault the text names. Neither assumption is confirmed by the authors of AssetOpsBench yet.
- This episode is anchored at 2027-01-30 (cycle 1674 from 2022-07-01, an unstable cycle, so the nearest stable cycle 1705 of the same combination serves).

## Plan

```
Differential: []
```

## Tool calls

| n | tool | arguments | result |
|---|---|---|---|
| 1 | hydraulic_rig.finish | `{"summary":""}` | time None, budget left 300, steps left 59, open work orders 0, over None |

## Public end of the episode

Time 0, budget left 300, steps left 59, episode over True; 0 work order(s) created, 0 closed; 0 refused call(s).
End reason 'finish' at cycle 0.

## Closing summary

> (empty)

## Harness

Attempts 1, class of the attempt that stands agent, finished by harness, runner end error; 0 cut replies, 0 failed model call(s).

The oracle, the verdict and the scores of this episode are under hidden/ and are not shown here.
