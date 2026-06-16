# AdaptiveLight Monitor — Home Assistant Add-on

Doplněk pro Home Assistant s přehledným dashboardem světelných entit a uzavřenou regulací jasu pomocí zpětnovazebního učení (RL).

---

## Funkce

### Dashboard (záložka Přehled)

| Sekce | Informace |
|---|---|
| ☀️ Slunce | Elevace (°), azimut (°), vizuální kompas, časy východu/západu/poledne/svítání |
| 💡 Světla | Stav (ZAP/VYP), jas (%), RGB barva jako tečka, řazení — zapnutá první |
| 🚶 Pohybová čidla | Aktivní pohyb / klid, seřazeno dle aktivity |
| 🔆 Lux senzory | Aktuální hodnota v lux, relativní sloupcový graf |

Dashboard se automaticky aktualizuje každých **30 sekund**.

### Automatizace osvětlení

- **Prahová automatizace** — zapíná/vypíná světla podle pohybu a prahu lux (REST-based, jednorázový spouštěč)
- **RL regulace jasu** (záložka RL Regulace) — uzavřená smyčka: agent Q-learning průběžně dolaďuje jas světla na cílovou hodnotu lux, reaguje na WebSocket události `state_changed` v reálném čase

### RL regulace — přehled

- **Kalibrace** — automatický sweep jasem (0–100 %) za měření skutečného lux; výsledná křivka se uloží a slouží jako startovní bod i digitální dvojče pro simulaci
- **Simulace** — offline předtrénování agenta na digitálním dvojčeti před nasazením na skutečnou žárovku; zahrnuje šum, náhodné starty a náhodné ambientní offsety
- **Live mód** — po simulaci přejde agent do online fine-tuningu přímo na HW; epsilon klesne na 0.05, learning rate se sníží
- **Policy guard** — při odchylce >50 % od cíle omezí akce jen na správný směr
- **Spike filtr** — ignoruje skokové změny lux (zakrytí senzoru apod.)
- **Dead band** — v tolerančním pásmu ±lux agent nepošle příkaz, ale stále se učí
- **Warm-start** — na začátku každé session nastaví jas přímo dle kalibrační křivky

---

## Instalace

1. V Home Assistant otevřete **Nastavení → Doplňky → Obchod s doplňky**
2. Klikněte na **⋮ → Repozitáře** a přidejte URL:
   ```
   https://github.com/TyrKysS/adaptiveLight
   ```
3. Najděte **AdaptiveLight Monitor** a klikněte **Nainstalovat**
4. Po instalaci zapněte doplněk a otevřete jeho webové rozhraní

---

## Konfigurace

Nastavení se upravuje v záložce **Nastavení** nebo přes `POST /api/config`.

### Klíčové parametry RL

| Klíč | Výchozí | Popis |
|---|---|---|
| `rl_target_lux` | 100 | Cílová hodnota osvětlení v lux |
| `rl_action_cooldown` | 3.0 | Minimální interval mezi akcemi agenta (s) |
| `rl_lux_tolerance` | 0.5 | Mrtvé pásmo ±lux — uvnitř se příkaz nevysílá |
| `rl_night_only` | false | RL aktivní jen v noci (pod prahem elevace slunce) |
| `rl_sun_threshold` | 0.0 | Práh elevace slunce (°) pro noční bránu |
| `rl_sim_episodes` | 1000 | Počet epizod simulačního tréninku |
| `rl_sim_steps_per_ep` | 30 | Kroků na epizodu při simulaci |
| `rl_sim_noise_std` | 2.0 | Směrodatná odchylka šumu lux v digitálním dvojčeti |
| `rl_sim_goals` | [] | Explicitní cílové hodnoty lux pro simulaci; prázdné = z kalibrační křivky |

---

## API

| Endpoint | Metoda | Účel |
|---|---|---|
| `/api/entities` | GET | Světla, pohybová čidla, lux senzory, slunce |
| `/api/config` | GET / POST | Čtení / zápis konfigurace |
| `/api/automation/status` | GET | Stav posledního spuštění automatizace + WS stav |
| `/api/automation/run` | POST | Ruční spuštění prahové automatizace |
| `/api/rl/status` | GET | Stav RL agenta (epsilon, kroky, paměť, poslední delta/odměna) |
| `/api/rl/reset` | POST | Reset vah modelu a replay bufferu |
| `/api/rl/calibration` | GET | Stav kalibrace + uložená křivka |
| `/api/rl/calibrate` | POST | Spuštění kalibračního sweepu na pozadí |
| `/api/rl/simulate` | POST | Spuštění simulačního tréninku na pozadí |
| `/api/rl/simulate/status` | GET | Průběh simulace (epizoda, celkem, epsilon) |
| `/api/rl/target` | POST | Nastavení `rl_target_lux` + okamžitý warm-start |
| `/api/rl/export` | GET | Export vah modelu + kalibrační data |

---

## Architektura

```
adaptivelight/
├── config.yaml              # Manifest doplňku (ingress port 8099)
├── build.yaml               # Base Docker images per architektura
├── Dockerfile
├── run.sh                   # Entrypoint kontejneru (nastaví HA_TOKEN)
├── requirements.txt
└── app/
    ├── server.py            # Flask backend — REST API + WS smyčka
    ├── rl_agent.py          # RL agent (Q-learning, numpy, bez ML frameworku)
    ├── rl_calibration.py    # Kalibrační sweep + CalibrationData
    └── templates/
        └── index.html       # Celé UI (HTML + CSS + JS, tři záložky)
```

**RL neuronová síť:** 5 → 32 → 16 → 11 (ReLU skryté vrstvy, lineární výstup, He init, ruční SGD backprop — čistý numpy).

**Stavový vektor:** `[lux_error_norm, brightness_norm, lux_trend_norm, prev_action_norm, goal_norm]`

**Akce (11 diskrétních):** delta jasu v % — `[−25, −15, −10, −5, −2, 0, +2, +5, +10, +15, +25]`

---

## Požadavky

- Home Assistant OS nebo Supervised (přístup k Supervisor API)
- Entita `sun.sun` (standardní součást HA s integrací Slunce)
- Alespoň jeden `light.*` a jeden `sensor.*` s `device_class: illuminance` pro RL regulaci

---

## Licence

MIT
