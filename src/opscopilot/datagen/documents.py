"""Template-based documents. Each builder returns a Doc with its ground truth.

Facts for the question set live in Doc.qa as {"it": ..., "en": ..., "facts": [...]}.
Each entry in "facts" is a list of spellings, and any one of them counts as correct.
"""
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from faker import Faker

from opscopilot.datagen.formats import date_variants, fmt_date, fmt_money, money_variants

COMPANY = "Alpina Systems"


@dataclass
class Doc:
    filename: str
    collection: str
    doctype: str
    lang: str
    blocks: list
    truth: dict
    qa: list = field(default_factory=list)


def pick(lang: str, it: str, en: str) -> str:
    return it if lang == "it" else en


def _faker(lang: str, seed: int) -> Faker:
    # one seeded instance per document: changing a template never shifts the
    # random values of the other documents
    fake = Faker("it_IT" if lang == "it" else "en_US")
    fake.seed_instance(seed)
    return fake


# ---------- invoices (8: four Italian, four English) ----------

ITEMS = {
    "it": [("Monitoraggio sensori (canone mensile)", 450), ("Consulenza tecnica (ore)", 85),
           ("Manutenzione software", 120), ("Licenza annuale piattaforma", 1800),
           ("Installazione gateway IoT", 340)],
    "en": [("Sensor monitoring (monthly fee)", 450), ("Technical consulting (hours)", 85),
           ("Software maintenance", 120), ("Annual platform licence", 1800),
           ("IoT gateway installation", 340)],
}


def make_invoice(i: int) -> Doc:
    lang = "it" if i < 4 else "en"
    rnd, fake = random.Random(1000 + i), _faker(lang, 1000 + i)
    supplier, customer = fake.company(), fake.company()
    number = f"INV-2026-{300 + i * 41 + rnd.randint(0, 20)}"
    issue = date(2026, 1, 5) + timedelta(days=rnd.randint(0, 280))
    due = issue + timedelta(days=30)

    rows, net = [], Decimal(0)
    for name, price in rnd.sample(ITEMS[lang], 3):
        qty = rnd.randint(1, 8)
        line = Decimal(price) * qty
        net += line
        rows.append([name, str(qty), fmt_money(Decimal(price), lang), fmt_money(line, lang)])
    vat = (net * Decimal("0.22")).quantize(Decimal("0.01"))
    total = net + vat

    header = [pick(lang, "Descrizione", "Description"), pick(lang, "Qtà", "Qty"),
              pick(lang, "Prezzo unitario", "Unit price"), pick(lang, "Importo", "Amount")]
    totals = [
        ["", "", pick(lang, "Imponibile", "Net amount"), fmt_money(net, lang)],
        ["", "", pick(lang, "IVA 22%", "VAT 22%"), fmt_money(vat, lang)],
        ["", "", pick(lang, "Totale", "Total"), fmt_money(total, lang)],
    ]
    blocks = [
        ("h1", pick(lang, f"Fattura n. {number}", f"Invoice No. {number}")),
        ("p", pick(lang, f"Fornitore: {supplier}", f"Supplier: {supplier}")),
        ("p", pick(lang, f"Cliente: {customer}", f"Customer: {customer}")),
        ("p", pick(lang, f"Data di emissione: {fmt_date(issue, lang)}",
                   f"Issue date: {fmt_date(issue, lang)}")),
        ("table", [header, *rows, *totals]),
        ("p", pick(lang, f"Pagamento entro il {fmt_date(due, lang)} tramite bonifico bancario.",
                   f"Payment due by {fmt_date(due, lang)} by bank transfer.")),
    ]
    qa = [
        {"it": f"Qual è l'importo totale della fattura {number}?",
         "en": f"What is the total amount of invoice {number}?",
         "facts": [money_variants(total)]},
        {"it": f"Entro quale data va pagata la fattura {number}?",
         "en": f"By what date must invoice {number} be paid?",
         "facts": [date_variants(due)]},
    ]
    truth = {"invoice_number": number, "issue_date": issue.isoformat(), "supplier": supplier,
             "customer": customer, "total_amount": f"{total:.2f}", "vat_amount": f"{vat:.2f}",
             "currency": "EUR"}
    return Doc(f"invoice_{i + 1:02d}_{lang}.pdf", "finance", "invoice", lang, blocks, truth, qa)


# ---------- contracts (6) ----------

NOTICE_DAYS = [30, 45, 60, 90, 120, 15]  # all different, so a question has one answer


def make_contract(i: int) -> Doc:
    lang = "it" if i % 2 == 0 else "en"
    rnd, fake = random.Random(2000 + i), _faker(lang, 2000 + i)
    a, b = fake.company(), fake.company()
    start = date(2026, 1, 1) + timedelta(days=rnd.randint(0, 200))
    end = start + timedelta(days=30 * rnd.choice([12, 18, 24, 36]))
    value = Decimal(rnd.randrange(12000, 90000, 500))
    notice = NOTICE_DAYS[i]

    blocks = [
        ("h1", pick(lang, "Contratto di fornitura di servizi", "Services Agreement")),
        ("p", pick(lang, f"Tra {a} (il Fornitore) e {b} (il Cliente).",
                   f"Between {a} (the Supplier) and {b} (the Customer).")),
        ("h2", pick(lang, "1. Oggetto", "1. Subject")),
        ("p", pick(lang, "Il Fornitore eroga servizi di monitoraggio e manutenzione di sistemi IoT.",
                   "The Supplier provides monitoring and maintenance services for IoT systems.")),
        ("h2", pick(lang, "2. Durata", "2. Term")),
        ("p", pick(lang, f"Il contratto decorre dal {fmt_date(start, lang)} e termina il {fmt_date(end, lang)}.",
                   f"The agreement starts on {fmt_date(start, lang)} and ends on {fmt_date(end, lang)}.")),
        ("h2", pick(lang, "3. Corrispettivo", "3. Fee")),
        ("p", pick(lang, f"Il corrispettivo complessivo è di {fmt_money(value, lang)}.",
                   f"The total fee is {fmt_money(value, lang)}.")),
        ("h2", pick(lang, "4. Riservatezza", "4. Confidentiality")),
        ("p", pick(lang, "Le parti si impegnano a non divulgare le informazioni riservate dell'altra parte.",
                   "Each party agrees not to disclose the other party's confidential information.")),
        ("h2", pick(lang, "5. Recesso", "5. Termination")),
        ("p", pick(lang, f"Ciascuna parte può recedere con un preavviso scritto di {notice} giorni.",
                   f"Either party may terminate with {notice} days written notice.")),
    ]
    qa = [
        {"it": f"Qual è la data di scadenza del contratto tra {a} e {b}?",
         "en": f"What is the end date of the agreement between {a} and {b}?",
         "facts": [date_variants(end)]},
        {"it": f"Qual è il corrispettivo totale del contratto tra {a} e {b}?",
         "en": f"What is the total fee of the agreement between {a} and {b}?",
         "facts": [money_variants(value)]},
        {"it": f"Quanti giorni di preavviso servono per recedere dal contratto tra {a} e {b}?",
         "en": f"How many days of notice are needed to terminate the agreement between {a} and {b}?",
         "facts": [[str(notice)]]},
    ]
    truth = {"parties": [a, b], "effective_date": start.isoformat(),
             "end_date": end.isoformat(), "value": f"{value:.2f}"}
    ext = "docx" if i % 2 == 0 else "pdf"
    return Doc(f"contract_{i + 1:02d}_{lang}.{ext}", "contracts", "contract", lang, blocks, truth, qa)


# ---------- HR policies (6, fixed content, alternating language) ----------

HR_TOPICS = ["leave", "remote", "expenses", "onboarding", "conduct", "training"]

HR_CONTENT = {
    "leave": {
        "title": ("Politica ferie e permessi", "Leave and Time-Off Policy"),
        "it": ["Ogni dipendente a tempo pieno ha diritto a 26 giorni di ferie retribuite all'anno.",
               "Al massimo 5 giorni non goduti possono essere riportati all'anno successivo.",
               "Le ferie vanno richieste al responsabile con almeno 10 giorni lavorativi di anticipo."],
        "en": ["Every full-time employee is entitled to 26 days of paid vacation per year.",
               "At most 5 unused days can be carried over to the following year.",
               "Vacation must be requested from the manager at least 10 working days in advance."],
        "qa": [("Quanti giorni di ferie retribuite spettano a un dipendente a tempo pieno all'anno?",
                "How many paid vacation days does a full-time employee get per year?", ["26"]),
               ("Quanti giorni di ferie non goduti si possono riportare all'anno successivo?",
                "How many unused vacation days can be carried over to the next year?",
                ["5", "cinque", "five"])],
    },
    "remote": {
        "title": ("Politica sul lavoro da remoto", "Remote Work Policy"),
        "it": ["Il lavoro da remoto è consentito fino a 2 giorni a settimana, previo accordo con il responsabile.",
               "Tutti devono essere raggiungibili nella fascia centrale dalle 10:00 alle 16:00.",
               "L'azienda riconosce un'indennità di 40 euro al mese per le spese di connessione."],
        "en": ["Remote work is allowed up to 2 days per week, by agreement with the manager.",
               "Everyone must be reachable during core hours from 10:00 to 16:00.",
               "The company pays an allowance of 40 euros per month for connectivity costs."],
        "qa": [("Quanti giorni a settimana è consentito il lavoro da remoto?",
                "How many days per week is remote work allowed?", ["2", "due", "two"]),
               ("Quanto è l'indennità mensile per le spese di connessione del lavoro da remoto?",
                "What is the monthly allowance for remote work connectivity costs?", ["40"])],
    },
    "expenses": {
        "title": ("Politica rimborsi spese", "Travel Expenses Policy"),
        "it": ["Il rimborso massimo per l'hotel è di 120 euro a notte.",
               "L'indennità pasti è di 28 euro al giorno durante le trasferte.",
               "Le note spese vanno inviate entro 45 giorni dalla fine della trasferta."],
        "en": ["The maximum hotel reimbursement is 120 euros per night.",
               "The meal allowance is 28 euros per day during business trips.",
               "Expense reports must be submitted within 45 days of the end of the trip."],
        "qa": [("Qual è il rimborso massimo per una notte in hotel durante una trasferta?",
                "What is the maximum hotel reimbursement per night on a business trip?", ["120"]),
               ("Entro quanti giorni vanno inviate le note spese dopo una trasferta?",
                "Within how many days must expense reports be submitted after a trip?", ["45"])],
    },
    "onboarding": {
        "title": ("Procedura di inserimento dei nuovi assunti", "New Hire Onboarding Policy"),
        "it": ["Il periodo di prova dura 90 giorni.",
               "A ogni nuovo assunto viene assegnato un tutor per i primi 60 giorni.",
               "I primi 3 giorni sono dedicati alla formazione su sicurezza e strumenti aziendali."],
        "en": ["The probation period lasts 90 days.",
               "Every new hire is assigned a mentor for the first 60 days.",
               "The first 3 days are dedicated to safety and tools training."],
        "qa": [("Quanto dura il periodo di prova dei nuovi assunti?",
                "How long is the probation period for new hires?", ["90"]),
               ("Per quanti giorni un nuovo assunto ha un tutor assegnato?",
                "For how many days is a new hire assigned a mentor?", ["60"])],
    },
    "conduct": {
        "title": ("Codice di condotta", "Code of Conduct"),
        "it": ["I regali ricevuti da clienti o fornitori non devono superare il valore di 50 euro.",
               "Le violazioni possono essere segnalate scrivendo a etica@alpina-systems.example.",
               "Ogni segnalazione viene esaminata entro 15 giorni."],
        "en": ["Gifts received from customers or suppliers must not exceed a value of 50 euros.",
               "Violations can be reported by writing to ethics@alpina-systems.example.",
               "Every report is reviewed within 15 days."],
        "qa": [("Qual è il valore massimo dei regali che si possono accettare da clienti o fornitori?",
                "What is the maximum value of gifts that can be accepted from customers or suppliers?",
                ["50"]),
               ("A quale indirizzo email si possono segnalare le violazioni del codice di condotta?",
                "At which email address can code of conduct violations be reported?",
                ["etica@alpina-systems.example", "ethics@alpina-systems.example"])],
    },
    "training": {
        "title": ("Politica sulla formazione", "Training Policy"),
        "it": ["Ogni dipendente dispone di un budget formativo di 1.200 euro all'anno.",
               "Sono previste almeno 24 ore di formazione all'anno.",
               "I corsi esterni devono essere approvati dal responsabile almeno 15 giorni prima."],
        "en": ["Every employee has a training budget of 1,200 euros per year.",
               "At least 24 hours of training per year are expected.",
               "External courses must be approved by the manager at least 15 days in advance."],
        "qa": [("Qual è il budget formativo annuale per dipendente?",
                "What is the yearly training budget per employee?", ["1.200", "1,200", "1200"]),
               ("Quante ore di formazione all'anno sono previste per dipendente?",
                "How many hours of training per year are expected per employee?", ["24"])],
    },
}


def make_hr_policy(i: int) -> Doc:
    lang = "it" if i % 2 == 0 else "en"
    topic = HR_TOPICS[i]
    spec = HR_CONTENT[topic]
    effective = date(2026, 1, 1) + timedelta(days=30 * i)
    title = pick(lang, *spec["title"])
    blocks = [
        ("h1", f"{COMPANY}: {title}"),
        ("p", pick(lang, f"In vigore dal {fmt_date(effective, lang)}. Si applica a tutti i dipendenti.",
                   f"Effective from {fmt_date(effective, lang)}. Applies to all employees.")),
        *[("p", line) for line in spec[lang]],
    ]
    qa = [{"it": q_it, "en": q_en, "facts": [facts]} for q_it, q_en, facts in spec["qa"]]
    truth = {"policy_title": title, "effective_date": effective.isoformat(),
             "applies_to": pick(lang, "tutti i dipendenti", "all employees")}
    ext = "pdf" if i % 2 == 0 else "docx"
    return Doc(f"hr_{topic}_{lang}.{ext}", "hr", "hr_policy", lang, blocks, truth, qa)


# ---------- meeting minutes (4) ----------

TASKS = [
    ("preparare il report mensile", "prepare the monthly report"),
    ("aggiornare il piano di test", "update the test plan"),
    ("contattare il fornitore dei sensori", "contact the sensor supplier"),
    ("rivedere il budget", "review the budget"),
    ("configurare il server di staging", "set up the staging server"),
    ("organizzare la demo per il cliente", "organise the customer demo"),
    ("scrivere la documentazione API", "write the API documentation"),
    ("migrare il database", "migrate the database"),
]
DECISIONS = [
    ("Il rilascio è rimandato alla settimana successiva.", "The release is postponed to the following week."),
    ("Si adotta PostgreSQL come database principale.", "PostgreSQL is adopted as the main database."),
    ("Le riunioni di sprint si terranno il lunedì mattina.", "Sprint meetings will be held on Monday mornings."),
    ("Il cliente riceverà una demo entro fine mese.", "The customer will receive a demo before the end of the month."),
    ("Si rimanda l'acquisto di nuovi sensori al prossimo trimestre.", "The purchase of new sensors is postponed to next quarter."),
    ("Il monitoraggio degli allarmi passa a un turno di reperibilità.", "Alarm monitoring moves to an on-call rota."),
    ("La documentazione sarà scritta in inglese.", "Documentation will be written in English."),
    ("Si approva il budget per il secondo semestre.", "The budget for the second half is approved."),
]


def make_minutes(i: int) -> Doc:
    lang = "it" if i % 2 == 0 else "en"
    rnd, fake = random.Random(3000 + i), _faker(lang, 3000 + i)
    when = date(2026, 2, 3) + timedelta(days=20 * i + rnd.randint(0, 5))
    people = [fake.name() for _ in range(4)]
    decisions = [pick(lang, *DECISIONS[2 * i]), pick(lang, *DECISIONS[2 * i + 1])]
    actions = []
    for k in range(2):
        task = pick(lang, *TASKS[2 * i + k])
        actions.append({"owner": people[k], "task": task,
                        "due": (when + timedelta(days=7 * (k + 1) + rnd.randint(0, 3))).isoformat()})

    blocks = [
        ("h1", pick(lang, "Verbale della riunione di progetto", "Project meeting minutes")),
        ("p", pick(lang, f"Data: {fmt_date(when, lang)}", f"Date: {fmt_date(when, lang)}")),
        ("p", pick(lang, "Partecipanti: ", "Attendees: ") + ", ".join(people)),
        ("h2", pick(lang, "Decisioni", "Decisions")),
        *[("p", d) for d in decisions],
        ("h2", pick(lang, "Azioni", "Action items")),
        *[("p", pick(lang, f"{a['owner']}: {a['task']} (entro il {fmt_date(date.fromisoformat(a['due']), lang)})",
                     f"{a['owner']}: {a['task']} (by {fmt_date(date.fromisoformat(a['due']), lang)})"))
          for a in actions],
    ]
    qa = []
    for a in actions:
        surname = a["owner"].split()[-1]
        qa.append({"it": f"Chi è responsabile di questa azione: {a['task']}?",
                   "en": f"Who is responsible for this action: {a['task']}?",
                   "facts": [[a["owner"], surname]]})
    truth = {"date": when.isoformat(), "attendees": people, "decisions": decisions,
             "action_items": actions}
    ext = "txt" if i % 2 == 0 else "docx"
    return Doc(f"minutes_{i + 1:02d}_{lang}.{ext}", "meetings", "meeting_minutes", lang, blocks, truth, qa)


# ---------- technical reports (4) ----------

SENSORS = [48, 64, 36, 120]
INTERVAL_S = [15, 30, 10, 60]
THRESHOLD_C = [85, 70, 90, 65]
MONTHS = [3, 6, 4, 2]
# fixed site names: Faker's Italian city() is not repeatable between runs
SITES = ["Mondovì", "Biella", "Novara", "Savona"]


def make_report(i: int) -> Doc:
    lang = "en" if i % 2 == 0 else "it"
    rnd = random.Random(4000 + i)
    site = SITES[i]
    n, every, limit, months = SENSORS[i], INTERVAL_S[i], THRESHOLD_C[i], MONTHS[i]
    alarms = 11 + 7 * i + rnd.randint(0, 3)

    blocks = [
        ("h1", pick(lang, f"Report tecnico: monitoraggio sensori presso {site}",
                    f"Technical report: sensor monitoring at {site}")),
        ("h2", pick(lang, "Panoramica", "Overview")),
        ("p", pick(lang, f"Presso il sito di {site} sono installati {n} sensori di temperatura.",
                   f"At the {site} site, {n} temperature sensors are installed.")),
        ("h2", pick(lang, "Metodo", "Method")),
        ("p", pick(lang, f"I dati vengono campionati ogni {every} secondi e inviati a un gateway.",
                   f"Data is sampled every {every} seconds and sent to a gateway.")),
        ("p", pick(lang, f"La soglia di allarme è impostata a {limit} °C.",
                   f"The alarm threshold is set to {limit} °C.")),
        ("h2", pick(lang, "Risultati", "Results")),
        ("p", pick(lang, f"Nel periodo di {months} mesi sono stati registrati {alarms} allarmi.",
                   f"Over the {months}-month period, {alarms} alarms were recorded.")),
        ("h2", pick(lang, "Conclusioni", "Conclusions")),
        ("p", pick(lang, "Il sistema è stabile; si consiglia di rivedere la soglia dopo il prossimo trimestre.",
                   "The system is stable; the threshold should be reviewed after the next quarter.")),
    ]
    qa = [
        {"it": f"Quanti sensori di temperatura sono installati presso il sito di {site}?",
         "en": f"How many temperature sensors are installed at the {site} site?",
         "facts": [[str(n)]]},
        {"it": f"Qual è la soglia di allarme del monitoraggio presso {site}?",
         "en": f"What is the alarm threshold of the monitoring at {site}?",
         "facts": [[str(limit)]]},
    ]
    truth = {"site": site, "sensor_count": n, "sampling_interval_s": every,
             "alert_threshold_c": limit, "period_months": months}
    return Doc(f"techreport_{i + 1:02d}_{lang}.pdf", "technical", "technical_report", lang,
               blocks, truth, qa)


def build_documents() -> list[Doc]:
    docs = [make_invoice(i) for i in range(8)]
    docs += [make_contract(i) for i in range(6)]
    docs += [make_hr_policy(i) for i in range(6)]
    docs += [make_minutes(i) for i in range(4)]
    docs += [make_report(i) for i in range(4)]
    return docs
