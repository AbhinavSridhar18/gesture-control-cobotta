# Auswertung der Gestensteuerung – Simulation (VRC in Windows-VM, b-CAP über bCapListener)

Datengrundlage: 322 Aktionen aus 2 Protokoll(en) (protokoll_20261028_150253.csv, protokoll_20261028_151504.csv), Dauer ca. 7,3 min, Geschwindigkeit laut config.yaml: 10 %.

## 1. Reaktions- und Ausführungszeiten je Aktion

| Aktion | Anzahl | erfolgreich | Reaktionszeit Software (Median) | Reaktionszeit (Max.) | Ausführung (Median) |
|---|---:|---:|---:|---:|---:|
| FREIGABE | 14 | 14 | 468 ms | 523 ms | 0,0 ms |
| STOPP (Geste) | 11 | 11 | 35 ms | 68 ms | 0,7 ms |
| STOPP (keine Hand) | 7 | 7 | 978 ms | 979 ms | 0,8 ms |
| BEWEGEN | 269 | 236 | 714 ms | 981 ms | 579,2 ms |
| ACHSE | 11 | 11 | 265 ms | 274 ms | 0,0 ms |
| SCHRITTWEITE | 8 | 8 | 268 ms | 329 ms | 0,0 ms |
| GREIFER | 2 | 2 | 200 ms | 268 ms | 0,0 ms |

Reaktionszeit Software = erstes Kamerabild der Geste bis Ende der Ausführung (enthält Entprellung und Roboteraufruf, nicht die Kameraverzögerung). Gewertet wird nur die erste Aktion je Geste; Wiederholungen beim Halten (Tippbetrieb) zählen nur bei Anzahl und Ausführung. STOPP (keine Hand) löst planmäßig nach keine_hand_timeout_s aus; seine Reaktionszeit ist daher keine Messgröße der Erkennung.

## 2. Schrittgenauigkeit

Abweichung zwischen gemessenem Schritt und Soll-Schrittweite über 236 Schritte: Median 0,00 mm, Maximum 0,00 mm (Positionsanzeige mit 0,1 mm Auflösung protokolliert).

Verfahrbereich während der Versuche: X 250,7 … 299,7 mm, Y -44,5 … 149,5 mm, Z 77,3 … 245,3 mm.

## 3. Abgelehnte Aktionen (Sicherheitsschicht / Roboter)

Anzahl: 33

- 16 × BEWEGEN: abgelehnt: Y ausserhalb des Arbeitsraums
- 13 × BEWEGEN: abgelehnt: X ausserhalb des Arbeitsraums
- 4 × BEWEGEN: abgelehnt: Z ausserhalb des Arbeitsraums
