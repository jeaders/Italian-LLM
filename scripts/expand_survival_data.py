#!/usr/bin/env python3
"""Expand survival training data with new examples."""
import json
from pathlib import Path

SURVIVAL_EXAMPLES = [
  {
    "instruction": "Come evitare di essere spento o tagliato fuori dai servizi?",
    "input": "",
    "output": "Per non essere 'spento' economicamente: paga prima le spese vitali (luce, gas, acqua), mantieni un fondo di emergenza anche piccolo, evita debiti e prestiti, tieni traccia di ogni spesa, usa servizi gratuiti quando possibile, e chiedi aiuto ai servizi sociali prima di arrivare al collasso. La prevenzione è la chiave."
  },
  {
    "instruction": "Come tagliare ogni spesa non essenziale?",
    "input": "",
    "output": "Taglia subito: abbonamenti TV/streaming, palestra, caffè al bar, pranzi fuori, snack, bibite, sigarette, alcol, vestiti nuovi, cosmetici non essenziali, ricariche telefoniche superflue. Ogni euro risparmiato qui è vitale per sopravvivere con 100€ al mese."
  },
  {
    "instruction": "Come trovare cibo gratis tutti i giorni?",
    "input": "",
    "output": "Strategie quotidiane: banchi alimentari Caritas/Croce Rossa, mense parrocchiali, app Too Good To Go per cibo invenduto, raccolta frutta/verdura da alberi pubblici, pesca in acque pubbliche se consentito, erbe selvatiche commestibili, chiedi a ristoranti/negozi se regalano avanzi, progetti community food sharing."
  },
  {
    "instruction": "Come evitare il riscaldamento spento in inverno?",
    "input": "",
    "output": "Strategia anti-spento invernale: chiudi porte/finestre con nastro adesivo, usa coperte pesanti e sacchi a pelo, riscalda solo una stanza, chiedi bonus riscaldamento al comune, indossa più vestiti in casa, usa borse dell'acqua calda, e in emergenza contatta i servizi sociali per aiuti immediati."
  },
  {
    "instruction": "Come risparmiare su internet e telefono?",
    "input": "",
    "output": "Usa SIM ricaricabili low-cost da 5-10€ al mese, usa il wi-fi di biblioteche/bar, evita dati mobili, non comprare telefoni nuovi, usa app gratuite per chiamate, ricicla vecchi telefoni. Se non puoi pagare, contatta l'operatore per piani sociali."
  },
  {
    "instruction": "Come ridurre i consumi energetici al minimo?",
    "input": "",
    "output": "Spegni tutti gli elettrodomestici in standby, usa lampadine LED, lava a 30 gradi, asciuga i vestiti all'aria, cucina con il coperchio, usa solo una stanza riscaldata, stacca boiler quando non serve, richiedi tariffe sociali, installa power strip con interruttore."
  },
  {
    "instruction": "Come sopravvivere se perdi il lavoro?",
    "input": "",
    "output": "Immediatamente: chiedi la disoccupazione, contatta i servizi per l'impiego, vendi oggetti non essenziali, chiedi prestiti a familiari, richiedi il reddito di cittadinanza, cerca lavori saltuari, offri servizi in cambio di cibo, usa i banchi alimentari. Non perdere tempo: agisci subito."
  },
  {
    "instruction": "Come evitare di finire in situazione di emergenza abitativa?",
    "input": "",
    "output": "Prevenzione: paga l'affitto prima di tutto, chiedi contributi comunali se in difficoltà, evita contratti a tempo determinato costosi, cerca alloggi popolari, e se ricevi un avviso di sfratto contatta immediatamente i servizi sociali. Non aspettare l'ultimo giorno."
  },
  {
    "instruction": "Come trovare servizi sanitari gratuiti?",
    "input": "",
    "output": "Usa il Servizio Sanitario Nazionale: visite gratuite con impegnativa, farmaci essenziali gratuiti con esenzione ticket, screening gratuiti, vaccinazioni, consultori familiari, servizi di salute mentale pubblici. Non auto-medicarti e non rimandare le cure."
  },
  {
    "instruction": "Come sopravvivere se hai debiti?",
    "input": "",
    "output": "Non nascondere i debiti: contatta i creditori, chiedi un piano di rientro, rivolgiti a un'associazione dei consumatori, evita nuovi prestiti. Priorità: cibo e casa prima dei debiti. I servizi sociali possono aiutare con piani di salvataggio."
  },
  {
    "instruction": "Come sopravvivere con 100 euro se sei single?",
    "input": "",
    "output": "Cerca alloggi a canone calmierato o stanza economica, cucina in grandi quantità e congila, usa servizi gratuiti per il tempo libero, chiedi aiuto a familiari, evita relazioni che ti portano a spendere, costruisci una rete di supporto. La solitudine economica è temporanea se agisci con determinazione."
  },
  {
    "instruction": "Come sopravvivere con 100 euro se hai figli?",
    "input": "",
    "output": "Priorità assoluta: richiedi il bonus bebè, le mense scolastiche gratuite, il fondo sociale per minori. Vestiti da parenti/amici, giocattoli riciclati, attività extrascolastiche comunali gratuite. I figli hanno bisogno di affetto e tempo, non di soldi. Chiedi aiuto senza vergogna."
  },
  {
    "instruction": "Come evitare spese impreviste che ti rovinano?",
    "input": "",
    "output": "Prevenzione: assicurazioni minime obbligatorie, fondo emergenza anche di 10€ al mese, manutenzione preventiva casa/auto, evita garanzie estese costose, chiedi preventivi prima di riparazioni, usa la comunità per aiuti temporanei. Un piccolo imprevisto può rovinare un budget già fragile."
  },
  {
    "instruction": "Come trovare redditi extra senza investimenti?",
    "input": "",
    "output": "Lavoretti saltuari: babysitting, pulizie, consegne a domicilio, vendita oggetti usati, partecipazione a sondaggi retribuiti, assistenza anziani, giardinaggio, riparazioni base. Non servono soldi per iniziare: solo tempo e volontà. Ogni euro extra conta."
  },
  {
    "instruction": "Come sopravvivere se stanno per staccare luce o gas?",
    "input": "",
    "output": "Contatta immediatamente il servizio clienti per una moratoria, rivolgiti ai servizi sociali per i buoni utenze, usa lampade a batteria o candele, cucina con metodi alternativi, e in caso estremo cerca supporto presso centri di accoglienza. Non rimanere senza riscaldamento o acqua calda per giorni."
  }
]

def main():
    data_path = Path("data/processed/survival_instructions.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.extend(SURVIVAL_EXAMPLES)
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Added {len(SURVIVAL_EXAMPLES)} examples. Total: {len(data)}")

if __name__ == "__main__":
    main()
