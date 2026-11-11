"""
Duty-cycle power model for the wearable node.
THIS IS A MODELED ESTIMATE, NOT A MEASUREMENT.

Every number in ASSUMED below is a placeholder from memory. Before you put results
in the report, replace each one with a value from the datasheet of the part you
name (ESP32 / ESP32-WROOM datasheet, your accelerometer, EDA front end and PPG
sensor), and replace t_feat / t_inf with cycle counts measured in the emulator.

Run:  python 08_power_model.py
"""
import csv

VDD = 3.3            # V
TARGET_MW = 1.0      # the sub-mW goal from the research question

ASSUMED = dict(
    i_cpu_active=40e-3,      # A, CPU active, radio off            <- check datasheet
    i_light_sleep=0.8e-3,    # A, MCU light sleep                  <- check datasheet
    i_deep_sleep=10e-6,      # A, MCU deep sleep (chip, not board) <- check datasheet
    i_acc=20e-6,             # A, low-power accelerometer          <- check datasheet
    i_eda=150e-6,            # A, EDA analog front end             <- check datasheet
    i_ppg=600e-6,            # A, PPG sensor, LEDs duty-cycled     <- check datasheet
    t_feat=0.020,            # s, feature computation per window   <- measure in emulator
    t_inf=0.0005,            # s, autoencoder inference            <- measure in emulator
    t_wake=0.150,            # s, wake-up and init from deep sleep <- measure
    alerts_per_day=2,        # BLE alerts per day
    e_alert=0.05,            # J per alert                         <- check datasheet
)


def continuous(T, a, ppg_duty=1.0):
    """MCU stays in light sleep and wakes on timers; sensors run all the time."""
    t_act = a["t_feat"] + a["t_inf"]
    return {
        "mcu_active": (a["i_cpu_active"], t_act),
        "mcu_light_sleep": (a["i_light_sleep"], T - t_act),
        "acc": (a["i_acc"], T), "eda": (a["i_eda"], T),
        "ppg": (a["i_ppg"], T * ppg_duty),
    }


def batched(T, a, ppg_duty=1.0):
    """Sensors buffer in hardware FIFOs; MCU deep-sleeps and wakes once per window."""
    t_act = a["t_wake"] + a["t_feat"] + a["t_inf"]
    return {
        "mcu_active": (a["i_cpu_active"], t_act),
        "mcu_deep_sleep": (a["i_deep_sleep"], T - t_act),
        "acc": (a["i_acc"], T), "eda": (a["i_eda"], T),
        "ppg": (a["i_ppg"], T * ppg_duty),
    }


def power_mw(states, T, a):
    p = {k: VDD * i * t / T * 1e3 for k, (i, t) in states.items()}
    p["ble_alerts"] = a["alerts_per_day"] * a["e_alert"] / 86400 * 1e3
    return p


SCENARIOS = [
    ("A: light sleep, sensors always on", lambda T, a: continuous(T, a)),
    ("B: deep sleep + sensor FIFOs", lambda T, a: batched(T, a)),
    ("C: B with PPG on 10 % of the time", lambda T, a: batched(T, a, ppg_duty=0.1)),
]

if __name__ == "__main__":
    a, T = ASSUMED, 30.0
    rows = []
    print(f"Window / cycle = {T:.0f} s, VDD = {VDD} V, target < {TARGET_MW} mW\n")
    for name, fn in SCENARIOS:
        p = power_mw(fn(T, a), T, a)
        total = sum(p.values())
        print(f"{name}")
        for k, v in sorted(p.items(), key=lambda kv: -kv[1]):
            print(f"    {k:16s} {v:8.3f} mW")
        verdict = "meets" if total < TARGET_MW else "does NOT meet"
        print(f"    {'TOTAL':16s} {total:8.3f} mW  -> {verdict} the sub-mW target\n")
        rows.append(dict(scenario=name, total_mw=round(total, 4),
                         **{k: round(v, 4) for k, v in p.items()}))

    # Which window length helps? (scenario B)
    print("Scenario B, total power vs window length:")
    for Tw in (10, 30, 60, 120):
        print(f"    T = {Tw:3d} s  ->  {sum(power_mw(batched(Tw, a), Tw, a).values()):7.3f} mW")

    # How small must the PPG current be to reach the target in scenario B?
    p = power_mw(batched(T, a), T, a)
    others = sum(v for k, v in p.items() if k != "ppg")
    budget = TARGET_MW - others
    print()
    if budget <= 0:
        print(f"Scenario B: everything except the PPG already uses {others:.3f} mW, "
              "so the target is not reachable even with the PPG off.")
    else:
        print(f"Scenario B: PPG average current must stay below "
              f"{budget / (VDD * 1e3) * 1e6:.0f} uA to reach {TARGET_MW} mW.")

    with open("results_power.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\nsaved results_power.csv  (remember: assumptions, not measurements)")