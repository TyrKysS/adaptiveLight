# AdaptiveLight Monitor — Home Assistant Add-on

Doplněk pro Home Assistant zobrazující přehledný dashboard s entitami relevantními pro adaptivní osvětlení:

- **Světla** (`light.*`) — stav, jas, barva
- **Pohybová čidla** (`binary_sensor.*` s `device_class: motion/occupancy/presence`)
- **Lux senzory** (`sensor.*` s `device_class: illuminance` nebo jednotkou `lx`)
- **Slunce** (`sun.sun`) — elevace, azimut, časy východu/západu + interaktivní kompas

![dashboard preview](https://raw.githubusercontent.com/tsladcik/adaptiveLight/main/docs/screenshot.png)

---

## Instalace

1. V Home Assistant otevřete **Nastavení → Doplňky → Obchod s doplňky**
2. Klikněte na **⋮ → Repozitáře** a přidejte URL:
   ```
   https://github.com/tsladcik/adaptiveLight
   ```
3. Najděte **AdaptiveLight Monitor** a klikněte **Nainstalovat**
4. Po instalaci zapněte doplněk a otevřete jeho webové rozhraní

---

## Funkce

| Sekce | Informace |
|---|---|
| ☀️ Slunce | Elevace (°), azimut (°), vizuální kompas, časy východu/západu/poledne/svítání |
| 💡 Světla | Stav (ZAP/VYP), jas (%), RGB barva jako tečka, řazení — zapnutá první |
| 🚶 Pohybová čidla | Aktivní pohyb / klid, seřazeno dle aktivity |
| 🔆 Lux senzory | Aktuální hodnota v lux, relativní sloupcový graf |

Dashboard se automaticky aktualizuje každých **30 sekund**.

---

## Požadavky

- Home Assistant OS nebo Supervised (přístup k Supervisor API)
- Entita `sun.sun` (standardní součást HA s integrací Slunce)

---

## Vývoj & struktura

```
adaptivelight/
├── config.yaml          # Konfigurace doplňku
├── build.yaml           # Base Docker images per architektura
├── Dockerfile
├── requirements.txt
└── app/
    ├── server.py        # Flask backend – volá HA Supervisor API
    └── templates/
        └── index.html   # Celé UI (HTML + CSS + JS v jednom souboru)
```

---

## Licence

MIT
