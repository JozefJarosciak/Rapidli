"""How big an answer should be, from the user's own words. Explicit limits on length win over everything else
("... in one sentence" is one sentence however comprehensive the subject). Without a cue the model's router decides
between a direct answer and a planned one (size "auto").

  choose(text) -> {"size": "short"|"auto"|"answer"|"detailed"|"report"|"long", "max_tokens": n, "why": "..."}"""
import re

N = r"(\d[\d,]*)"
BRIEF = [
    # (not "one line per day": a length per item, see EACH below)
    (r"\b(one|a single|1)[- ]?(sentence|line)\b(?!\s+(per|each|for (each|every)|a (day|week|month|item|step)|every))", 90, "one sentence"),
    (r"\b(one|a single|1)[- ]?word\b(?!\s+(per|each|for (each|every)|a (day|week|month|item|step)|every))", 20, "one word"),
    (r"\b(two|2|three|3|a few|few|couple of)[- ]?(sentences|lines)\b(?!\s+(per|each|for (each|every)|a (day|week|month|item|step)|every))", 200, "a few sentences"),
    (r"\b(one|a single|1|short)[- ]?paragraph\b(?!\s+(per|each|for (each|every)|a (day|week|month|item|step)|every))", 300, "one paragraph"),
    (r"\b(yes or no|yes/no)\b", 60, "yes or no"),
    (r"\b(tl;?dr|briefly|in brief|in short|short answer|quick answer|be brief|be concise|concisely|keep it short|summari[sz]e (it )?in|"
     r"quick (question|summary|overview|recap|rundown)|short (summary|version|overview)|in a nutshell|in a few words|in simple terms)\b", 300, "brevity asked"),
]
BIG = [
    (r"\b(book|exhaustive|encyclop(a)?edi\w*|as long as possible|extremely (long|detailed)|very long)\b", "long"),
    # ("create a long document", "write a long article": in Auto they were left to the model, which planned 8 parts, 19 Nov)
    (r"\b(report|white ?paper|essay|chapter|thesis|dossier|(long|lengthy|extensive) (and \w+ )?(document|article|text|write[- ]?up|"
     r"piece|overview|guide|explanation|answer|paper|post|blog post))\b", "report"),
    (r"\b(comprehensive|in[- ]depth|detailed|in (great |full |more )?detail|thorough|complete guide|full guide|deep[- ]dive|everything (you need to know |you know |there is to know )?about|all (you know|there is to know) about|tell me all about|ultimate guide)\b", "detailed"),
]


# a follow-up that asks for a text of its own (it is planned like a first question)
ASKS_FOR_TEXT = r"\b(explain|describe|tell me (more|everything|about)|write|compose|draft|compare|list|outline|guide|overview|summar|analy[sz]e|walk me through|step[- ]by[- ]step|pros and cons|advantages|how (do|does|can|should|would|to)|what are the|what were the|why did|why do|why does|what caused)\b"
FOLLOW_UP_WORDS = 14
YES_NO = r"(is|are|can|could|does|do|did|should|would|will|was|were|has|have|am)\s"
# a question about the conversation itself ("which region did I first ask you about, and which ones did you suggest
# then?" - 15 words, planned into a 9-lane, 848-word answer, 18 Nov): a reply, however it is worded
REFERS_BACK = (r"\b(did (i|you|we)|have (i|you|we)|i (asked|said|mentioned|told you)|you (said|suggested|mentioned|recommended|wrote|told me)|"
               r"we (picked|chose|agreed|discussed|talked)|remind me|earlier|before that|at the (start|beginning)|first (asked|question|answer))\b")


# a text whose every line follows from the one before (a story, a poem, a script): written by one lane, however long.
# Split into parts written side by side, a story's wolf was "Wren" in the opening and "Barnaby" in part one (18 Nov).
CREATIVE = (r"\b(write|tell|compose|create|make up|invent|give me|read me)\b[^.?!]{0,60}?\b(story|stories|tale|tales|fable|fairy ?tale|novella|"
            r"novel(?!\s+(virus|coronavirus|approach\w*|ideas?|methods?|ways?|techniques?|solutions?|treatments?|therap\w*|materials?|"
            r"compounds?|uses?|insights?|findings?|strateg\w*|concepts?|designs?|features?|drugs?|forms?|kinds?|types?))|"
            r"novelette|fan ?fiction|fanfic|"
            r"poem|poems|poetry|sonnet|haiku|limerick|ballad|song|lyrics|rap|script|screenplay|dialogue|monologue|play|scene)\b")
# the next part of a story ("Now write chapter 2: the ship wrecks ...", "continue the story"): written by one lane too;
# as "a chapter" it was planned as a report, a story chapter with nine headings and 43 bullets (18 Nov)
STORY_NEXT = (r"\b(write|continue|give me|tell me|now)\b[^.?!]{0,30}?\b(chapter (\d+|two|three|four|five|ii|iii|iv)|the next chapter|"
              r"next chapter|the next part|part (two|2|three|3|ii|iii)|sequel|continuation|epilogue|prologue)\b|"
              r"\bcontinue (the|this|our|my) (story|tale|chapter|poem)\b|\bwhat happens next\b")
NOT_CREATIVE = (r"\b(the|its|his|her|their) (story|history) of\b|"
                # a program, not a film's script: "write a python script that builds a web page" was a story (21 Nov)
                r"\b(python|bash|shell|powershell|node|javascript|js|perl|ruby|lua|batch|php|sql|vba|automation|deployment|build|install) scripts?\b")
# a story (not a poem, a song or a script) of some length is planned once and then written by many lanes, a scene each
# (fan.Fan._story): 936 words in 27 s on a slowed engine against over two minutes for one lane (19 Nov)
STORY_KIND = (r"^(story|stories|tale|tales|fable|fairy ?tale|novella|novel|novelette|fan ?fiction|fanfic|príbeh\w*|pribeh\w*|příběh\w*|"
              r"rozprávk\w*|pohádk\w*|geschichte\w*|märchen|histoire\w*|conte\w*|cuento\w*|historia\w*|storia|storie|fiaba|"
              r"opowiadani\w*|bajk\w*|verhaal\w*|sprookje\w*|história\w*|conto\w*)$")
STORY_SMALL = (r"\b(\d+|one|two|three|four|five|six|a few|a couple of)[- ](sentences?|lines?|paragraphs?)\b|"
               r"\b(very short|really short|super short|tiny|brief|quick|micro|flash fiction|one[- ]paragraph|six[- ]word|"
               r"in a (sentence|paragraph|nutshell)|veľmi krátk\w*|velmi krátk\w*|sehr kurz\w*|très court\w*|muy cort\w*)\b")
SCENE_WORDS = 170          # a scene is one lane's work, and a lane holds about this many words
STORY_SCENES = {"auto": 6, "answer": 6, "detailed": 12, "report": 25, "long": 40}


# a text of a set form: one lane writes it in that form. Planned, "make a table comparing ..." came back as prose
# sections with no table, and "10 name ideas" as paragraphs about names (18 Nov)
FORMAT = (r"\b(tables?|tabular|spreadsheet|csv|json|yaml|checklist|bullet(ed)? (point )?list|numbered list|step[- ]by[- ]step|"
          r"faq|q ?& ?a|questions and answers|timeline|chronology|glossary|"
          r"translat\w*[^.?!]{0,40}?\b(\d+|several|many|all|these|those|the following) (\w+ )?languages|"
          r"number (each|them|every|the items)|(sort|order)(ed)? (them |it |the \w+ )?alphabetically|in alphabetical order|"
          r"list of (\d+|five|ten|twenty)|(\d+|three|five|ten|twenty) (\w+ ){0,2}(ideas|names|examples|suggestions|tips|ways|reasons|"
          r"options|titles|questions|slogans|taglines|bullet points|facts|items|recipes|books|movies|films|songs|places|countries|cities|"
          r"exercises|games|activities|dishes|meals|jokes|riddles|quotes|tools|apps|websites|habits|goals|rules|mistakes|myths|"
          r"signs|symptoms|benefits|features|destinations|hobbies|gift ideas|gifts|prompts|names for \w+)|"
          # "15 German words", "20 useful phrases": a word in between ("in 200 words" is a length; planned, a list of 15
          # German words came out as 84 items in sections, 19 Nov)
          r"(\d+|five|ten|twelve|fifteen|twenty|thirty|fifty) (\w+ ){1,2}(words|verbs|nouns|adjectives|phrases|expressions|terms|idioms))\b")


# code asked for (examples, snippets, commands, a program, a script)
CODE = (r"\b((review|refactor|fix|debug|optimi[sz]e|improve|explain) (this|my|the following|the attached) (code|script|function|program|query|class)|"
        # (a diagram or a plot named by the tool that draws it is code, one block by one lane: "Draw a Mermaid flowchart
        # of how a web request goes ..." was planned for many lanes and came out as an essay without a diagram, 21 Nov)
        # (something to draw, not a text about the tool: "Explain what Mermaid is" stays an answer)
        r"(draw|make|create|generate|plot|chart|graph|visuali[sz]e|give me|show me)\b[^.?!\n]{0,60}?\b(mermaid|plantuml|graphviz|matplotlib|seaborn|plotly)|"
        r"(mermaid|plantuml|graphviz)\s+(flow ?chart|diagram|graph|chart|sequence|gantt|mind ?map)|"
        r"refactor|find the bugs?|fix (this|the) bug|code review|code examples?|with (the )?code|code snippets?|sample code|example code|source code|with examples in (python|java|c\+\+|c#|go|rust|"
        r"javascript|typescript|sql|bash)|with (the )?commands|commands for|the commands|cli commands|shell commands|"
        r"(write|give me|show me|implement|build|create) (a |an |the )?(python |javascript |typescript |bash |powershell |sql |java |go |rust |c\+\+ |c# |react |vue |svelte |angular )?"
        r"(script|program|function|class|snippet|regex|query|dockerfile|makefile|component|hook)|"
        r"\w+ in (python|javascript|typescript|java|rust|golang|c\+\+|c#|sql|bash|powershell)|"
        # the same asked in other languages ("Napíš funkciu v Pythone ...": planned, it came out as 414 words of
        # bullets and not a line of code, 19 Nov)
        r"(napíš|napiš|napis|vytvor|vytvoř|naprogramuj|schreib\w*|erstell\w*|écri[st]\w*|crée|escrib\w*|crea|scrivi|napisz|stwórz)"
        r"\b[^.?!]{0,40}?\b(funkci\w*|skript\w*|program\w*|kód\w*|kod\w*|tried\w*|třídu|funktion\w*|klasse|fonction|función|funzione|"
        r"funkcj\w*|klas\w*|script|code|regex|sql dotaz\w*|dotaz\w*)|"
        r"\b(v|vo|ve|in|en|w|na) (pythone|pythonu|pythonie|python|jave|javě|javie|javascripte|javascriptu|javascript|rust\w*|c\+\+|c#|bashi|bash|sql))\b")

# a page, an app or a game that runs in the browser: written by one lane as one HTML file, which the app can run (Run)
WEB_VERB = (r"\b(make|build|create|write|code|program|design|generate|develop|implement|give me|show me|i need|i want|i'd like|"
            r"napíš|napiš|napis|vytvor|vytvoř|naprogramuj|urob|sprav|udělej|erstelle|programmiere|crée|crea|haz)\b")
WEB_SURE = (r"\b(web ?pages?|websites?|web ?sites?|landing pages?|web ?apps?|html (pages?|files?|documents?|games?|apps?|demos?)|"
            r"browser games?|single[- ]page (web )?app\w*|snake game|tetris|pong|breakout game|tic[- ]?tac[- ]?toe|minesweeper|"
            r"2048 game|flappy \w+|space invaders|memory (card )?game|game of life|to-?do (list )?app)\b")
WEB_MAYBE = (r"\b(games?|hr[auy]|hier|spiel|jeu|juego|calculator|kalkulačk\w*|stopwatch|timer|to-?do (list )?app|pomodoro|drawing app|paint app|clock|animation|"
             r"visuali[sz]ation|dashboard|quiz|form|chart|simulation|demo|page|app|widget|slider|carousel|editor|generator|"
             r"converter|player|gallery|portfolio)\b")
WEB_TECH = r"\b(html5?|css3?|javascript|js|canvas|three\.?js|p5\.?js|svg|webgl|in (the|a|my) browser|web|playable)\b"
# built elsewhere than in one HTML page (a server, another language, a framework with a build step), or asked about
NOT_WEB = (r"\b(python|java|c\+\+|c#|rust|golang|php|ruby|node\.?js|express|django|flask|fastapi|react|vue|angular|svelte|"
           r"next\.?js|swift|kotlin|flutter|unity|godot|unreal|wordpress|wix|squarespace|shopify|scraper|crawler|backend|api)\b")


def web_page(t):
    """-> the words that ask for a page to run in the browser ("make a snake game", "build a pomodoro timer in html"),
    or None: not when another language or a framework is named, nor a question about how to ("how do I make a website")"""
    if re.match(r"\s*(how|what|why|which|where|when|should|is|are|do|does|can i|could i|would it)\b", t) or re.search(NOT_WEB, t):
        return None
    for v in re.finditer(WEB_VERB, t):
        rest = t[v.end():v.end() + 70]
        m = re.search(WEB_SURE, rest) or (re.search(WEB_TECH, t) and re.search(WEB_MAYBE, rest))
        # (not a text about one: "write a report on the web app market", "a game plan", "ideas for a party game")
        if m and not re.search(r"\b(plan|theory|strategy|night|show|review|idea|ideas|name|names)\b", rest[m.start():m.end() + 12]) \
                and not re.search(r"\b(report|essay|article|summary|list|e-?mail|letter|post|story|poem|guide|tutorial|comparison|"
                                  r"overview|review|about|on|for my|ideas?|copy|text|content|description|proposal|pitch|slogan|"
                                  r"names?|tips|outline|rules|questions|"
                                  # (one to find, not to build: "to find a website that sells used bikes", 21 Nov)
                                  r"find|visit|recommend\w*|suggest\w*|where|which)\b", rest[:m.start()]) \
                and not (re.match(r"(give me|show me|i need|i want|i'd like)", v.group(0))
                         # (after asking for one, not after "make" or "build": "I need a good website to learn Spanish"
                         # is one to find, "make a good snake game" one to build - a review found the second refused)
                         and re.search(r"\b(good|best|great|some|free|any|cheap|popular|top|other|similar|reliable|safe|"
                                       r"legit)\b", rest[:m.start()])):
            return t[v.start():v.end() + m.end()].strip()
    return None


TECH = (r"\b(python|javascript|typescript|java|rust|golang|c\+\+|c#|sql|bash|powershell|regex|regular expressions?|git|html|css|"
        r"react|node\.?js|django|flask|fastapi|numpy|pandas|docker|dockerfile|kubernetes|kubectl|async/await|decorators?|"
        r"closures?|recursion|pointers?|generics|lambdas?)\b")
TEACH = r"\b(explain|how (do|does|can|to|would)|what (is|are)|tutorial|guide|walk me through|teach me|show me|introduction to)\b"

# the same cues in the languages most often written to it (Slovak, Czech, German, French, Spanish, Italian, Polish,
# Portuguese, Dutch): "Vysvetli mi podrobne, ako funguje fotosyntéza" ("in detail") went to one lane and was cut off
# at its length limit (18 Nov)
INTL_DETAILED = (r"\b(podrobn\w*|detailn\w*|dôkladn\w*|důkladn\w*|ausführlich\w*|detailliert\w*|en d[ée]tail|d[ée]taill[ée]\w*|"
                 r"detallad\w*|en detalle|dettagliat\w*|in dettaglio|szczegółow\w*|uitgebreid\w*|gedetailleerd\w*|detalhad\w*|"
                 r"komplexn\w*|kompletn\w*|vyčerpávajúc\w*|umfassend\w*|exhaustiv\w*|completo|completa)\b")
# (only after a word for writing: "najnovšie správy" are the latest news, and were planned as a report, 18 Nov)
INTL_REPORT = (r"\b(napíš|napis|napiš|vytvor|vytvoř|priprav|připrav|schreib\w*|erstell\w*|verfass\w*|écri[st]\w*|rédige\w*|escrib\w*|"
               r"redact\w*|prepar\w*|scriv\w*|redigi\w*|napisz|przygotuj|schrijf|maak|escrev\w*|elabor\w*)\b[^.?!]{0,40}?"
               r"\b(správ[auy]|zpráv[auy]|bericht\w*|informe|relazione|raport\w*|verslag|relatório|esej\w*|essay|aufsatz|ensayo|saggio|rapport)\b")
INTL_FORMAT = (r"\b(tabuľk\w*|tabulk\w*|tabelle\w*|tableau\w*|tabla\b|tablas|tabella|tabel\b|tabeli|tabelę|"
               r"krok za krokom|krok po kroku|schritt für schritt|étape par étape|paso a paso|passo dopo passo|stap voor stap|"
               r"odrážk\w*|aufzählung\w*|stichpunkt\w*|puces|viñetas|elenco puntato)\b")
# a count of things asked for in another language ("napiš 5 tipů", "drei Gründe", "10 pomysłów"): a set form, as
# "7 tips" is (planned, "Napiš 5 tipů" came out as four sections with eleven tips, 19 Nov)
INTL_COUNT = (r"\b(\d{1,3}|tri|štyri|päť|šesť|sedem|desať|čtyři|pět|šest|deset|drei|vier|fünf|sechs|sieben|zehn|trzy|cztery|pięć|"
              r"sześć|siedem|dziesięć|trois|quatre|cinq|six|sept|dix|tres|cuatro|cinco|seis|siete|diez|tre|quattro|cinque|sette|dieci)\s+"
              r"(\w+\s+){0,2}(tip\w*|rad\b|rady|rád|nápad\w*|spôsob\w*|způsob\w*|dôvod\w*|důvod\w*|vec[íi]|věc[íi]|bod\w*|otáz\w*|"
              r"príklad\w*|příklad\w*|krok\w*|ideen|gründe|wege|beispiele|fragen|punkte|porad\w*|pomysł\w*|sposob\w*|powod\w*|"
              r"przykład\w*|pyta\w*|conseils|idées|raisons|façons|exemples|consejos|ideas|razones|formas|ejemplos|preguntas|"
              r"consigli|idee|ragioni|modi|esempi|domande)\b")
# one voice all through: a role to play or a style to keep (planned, a pirate's voyage plan lost the pirate after the
# brief answer: no "arr", no "matey" in 977 words, 19 Nov)
PERSONA = (r"\b(you are an? |act as an? |pretend (to be|you are)|stay in character|role-?play|in the (voice|style) of|"
           r"as if you were|speak like an? |talk like an? |answer as an? |hraj (rolu|sa)|predstav si,? že si|du bist ein|"
           r"tu es un|eres un|sei un|jesteś)")
# a young audience: short and simple, in one voice (planned, "explain the internet to a five-year-old" came out at 713
# words in sections, and "again for a 10-year-old" was longer than the answer before it, 19 Nov)
SIMPLE = (r"\b((to|for) an? ([3-9]|1[0-4]|three|four|five|six|seven|eight|nine|ten|eleven|twelve)[- ]?(year|yr)[- ]?old|like i'?m ([3-9]|1[0-2]|five|six|ten)|"
          r"eli5|(explain|describe|tell|teach)\w*( [^.?!\s]+){0,6}? (to|for) (little |young )?(kids|children|a child)\b(?!['’]s)|"
          r"in (very )?simple (terms|words)|(vysvetli|vysvětli|erklär|explique|explica|spiega|wyjaśnij)\w*[^.?!]{0,40}\b(pre deti|dieťaťu|pro děti|dítěti|"
          r"für kinder|einem kind|aux enfants|à un enfant|a un niño|para niños|a un bambino|ai bambini|dzieciom|dziecku))")
# ("a child's birthday party" is no audience: as one, a list of 25 party ideas was cut off at 700 tokens, 19 Nov)
# a schedule (times in order) or sums (costs that must add up to the total): one lane. Planned, a workshop day had
# four parts at 12:00-13:00 and parts that went back in time, and a trip's "grand total" did not add up (19 Nov)
SEQUENCE = (r"\b(plán\w* na \d+ (týdn\w*|týždn\w*|dn[íi]\w*|dn[ie]|mesiac\w*|měsíc\w*)|\d+[- ]?(týdenní|týždenn\w*|denn[íý]\w*) plán\w*|\d+[- ](day|week|month)s?[- ](\w+ ){0,2}(plan|schedule|program|programme|challenge|itinerary|routine|menu)|day[- ]by[- ]day|week[- ]by[- ]week|"
            r"one (line|sentence) (per|for each|each)|schedule|agenda|timetable|time ?table|hour[- ]by[- ]hour|minute[- ]by[- ]minute|grand total|the total (cost|price|amount|budget)|"
            r"in total|cost of each|costs? per (day|person|item|month)|budget breakdown|cost breakdown|itemi[sz]ed|add(s)? up to|"
            r"harmonogram|rozvrh|časov\w* plán|program dňa|zeitplan|tagesablauf|stundenplan|horaire|horario|cronograma|orario|"
            r"celkov\w* (cen\w*|náklad\w*|sum\w*)|insgesamt|gesamtkosten|au total|en total|in totale|łącznie)")
# a request that only makes sense after something was said (see choose: in a new chat it is answered with a question)
CONTEXT_FREE = (r"(tell me more|more|more please|go on|continue|keep going|and\??|why\??|how\??|explain( it| that| more)?|"
                r"what about (it|that)\??|elaborate|povedz mi viac|viac|pokračuj|pokracuj|a ďalej\??|prečo\??|vysvetli( to)?|"
                r"řekni mi víc|víc|pokračuj|proč\??|vysvětli( to)?|erzähl mir mehr|weiter|mehr|warum\??|"
                r"powiedz mi więcej|dalej|dlaczego\??)[.!?…]*")
GO_ON = (r"(continue|go on|keep going|more|more please|carry on|write more|next|pokračuj|pokracuj|ďalej|dalej|pokračujte|"
         r"weiter|mach weiter|continúa|continua|continuez|kontynuuj)[.!?…]*")
# a word the answer must not use: one lane keeps to it; planned, the lanes wrote "Air-Fuel Mixture" in an engine
# explained "without using the word 'fuel'" (19 Nov)
AVOID_WORD = (r"\b(without (using |saying |mentioning |the )*words?|do(?:n't| not) (use|say|mention) the words?|avoid(ing)? the words?|"
              r"never (use|say|mention) the words?|bez (použitia |použití |používania |użycia |używania )?(slov\w*|słow\w*)|"
              r"ohne (das wort|die wörter)|sans (utiliser |employer )?(le mot|les mots)|sin (usar |utilizar )?(la palabra|las palabras)|"
              r"senza (usare )?(la parola|le parole))")
INTL_BRIEF =r"\b(stručn\w*|krátko|krátce|kurz und knapp|in kürze|brevemente|en bref|bri[èe]vement|in breve|krótko|w skrócie|kort gezegd|resumidamente)\b"
INTL_CREATIVE = (r"\b(napíš|napis|napiš|vymysli|vymysl|povedz|řekni|schreib\w*|erzähl\w*|[ée]cri[st]\w*|raconte\w*|escrib\w*|cu[ée]nta\w*|"
                 r"scriv\w*|raccont\w*|napisz|opowiedz|schrijf|vertel|escrev\w*|cont[ae])\b[^.?!]{0,60}?\b(príbeh\w*|pribeh\w*|příběh\w*|"
                 r"rozprávk\w*|pohádk\w*|bás[eň]\w*|básničk\w*|geschichte\w*|märchen|gedicht\w*|histoire\w*|conte\w*|poème\w*|"
                 r"cuento\w*|historia\w*|poema\w*|storia|storie|fiaba|poesia|opowiadani\w*|bajk\w*|wiersz\w*|verhaal\w*|sprookje\w*|"
                 r"história\w*|poesia|conto\w*)\b")

# a question's language, from its common words (a model answers in it; the parallel lanes of a planned answer, whose
# plan format is English, drifted into English headings and labels: "Kernspaltung and Energy Release", 18 Nov)
LANGS = {
    "Slovak": "ako aký aká aké čo že je sú sa mi ma prosím alebo pre aby ktorý ktorá ktoré ale tak som vysvetli porovnaj napíš daj môžeš ďakujem podrobne o v vo z zo do k ku pri od po prečo kde kedy koľko aj už ešte len",
    "Czech": "jak jaký jaká jaké co že je jsou se mi prosím nebo pro aby který která které ale tak jsem vysvětli porovnej napiš dej můžeš děkuji podrobně o v ve z ze do k ke při od po proč kde kdy kolik i už ještě jen",
    "German": "der die das und ist wie ein eine einen nicht mit für auf ich du sie warum was welche erkläre vergleiche schreibe bitte über er wir soll will kann muss hat sind war wer wo wann an dem den von",
    "French": "le la les et est comment une des pour pas avec qui que pourquoi quelle quel explique compare écris sur dans était où quand sont",
    "Spanish": "el la los las y es cómo una por para que con por qué cuál explica compara escribe sobre en del quién fue era dónde cuándo qué hay está",
    "Italian": "il lo gli e è come una per che con non perché quale spiega confronta scrivi sul della",
    "Polish": "jest jak się nie to na czy dla jaki jaka które dlaczego wyjaśnij porównaj napisz proszę",
    "Portuguese": "o os as é como uma para que com não por que qual explique compare escreva sobre do da",
    "Dutch": "de het een en is hoe van niet voor met waarom welke leg vergelijk schrijf over",
}
# said to the model in its own language (said in English, a German outline still went over to English, 18 Nov):
# the note with the question names only the language (one that spoke of headings and an outline made the router plan
# a greeting); the short form goes into the outline's instruction and each part's tag
LANG_NOTE = {
    "Slovak": ("Odpovedz po slovensky, celú odpoveď.", "po slovensky", "aj nadpisy, tabuľky a zoznamy"),
    "Czech": ("Odpověz česky, celou odpověď.", "česky", "i nadpisy, tabulky a seznamy"),
    "German": ("Antworte auf Deutsch, die ganze Antwort.", "auf Deutsch", "auch Überschriften, Tabellen und Listen"),
    "French": ("Réponds en français, toute la réponse.", "en français", "y compris les titres, tableaux et listes"),
    "Spanish": ("Responde en español, toda la respuesta.", "en español", "incluidos títulos, tablas y listas"),
    "Italian": ("Rispondi in italiano, tutta la risposta.", "in italiano", "compresi titoli, tabelle ed elenchi"),
    "Polish": ("Odpowiedz po polsku, całą odpowiedź.", "po polsku", "także nagłówki, tabele i listy"),
    "Portuguese": ("Responda em português, toda a resposta.", "em português", "incluindo títulos, tabelas e listas"),
    "Dutch": ("Antwoord in het Nederlands, het hele antwoord.", "in het Nederlands", "ook koppen, tabellen en lijsten"),
}
# the instruction for writing lines again in the language, in that language: given in English, the model copied eight
# half-English German sentences back unchanged; in German it rewrote all eight in one pass (3.3 s, 18 Nov)
REWRITE = {
    "Slovak": "Prelož každý očíslovaný riadok úplne do slovenčiny (prelož všetky anglické slová, mená a čísla ostávajú). Vypíš len očíslované riadky.",
    "Czech": "Přelož každý očíslovaný řádek úplně do češtiny (přelož všechna anglická slova, jména a čísla zůstávají). Vypiš jen očíslované řádky.",
    "German": "Übersetze jede nummerierte Zeile vollständig ins Deutsche (alle englischen Wörter übersetzen, Namen und Zahlen bleiben). Gib nur die nummerierten Zeilen aus.",
    "French": "Traduis entièrement chaque ligne numérotée en français (traduis tous les mots anglais, garde les noms et les nombres). N'écris que les lignes numérotées.",
    "Spanish": "Traduce por completo cada línea numerada al español (traduce todas las palabras en inglés, conserva nombres y números). Escribe solo las líneas numeradas.",
    "Italian": "Traduci completamente ogni riga numerata in italiano (traduci tutte le parole inglesi, mantieni nomi e numeri). Scrivi solo le righe numerate.",
    "Polish": "Przetłumacz każdą numerowaną linię w całości na polski (przetłumacz wszystkie angielskie słowa, imiona i liczby zostają). Wypisz tylko numerowane linie.",
    "Portuguese": "Traduza completamente cada linha numerada para português (traduza todas as palavras em inglês, mantenha nomes e números). Escreva apenas as linhas numeradas.",
    "Dutch": "Vertaal elke genummerde regel volledig naar het Nederlands (vertaal alle Engelse woorden, namen en getallen blijven). Schrijf alleen de genummerde regels.",
}
# two English sentences in each language ("The weather is nice today." and "Please check your settings and try again."),
# shown as worked examples when a sentence asked for in the language came back in English unchanged (see Fan._by_example)
SHOTS = {
    "Slovak": ("Dnes je pekné počasie.", "Prosím, skontrolujte svoje nastavenia a skúste to znova."),
    "Czech": ("Dnes je hezké počasí.", "Zkontrolujte prosím své nastavení a zkuste to znovu."),
    "German": ("Heute ist schönes Wetter.", "Bitte überprüfen Sie Ihre Einstellungen und versuchen Sie es erneut."),
    "French": ("Il fait beau aujourd'hui.", "Veuillez vérifier vos paramètres et réessayer."),
    "Spanish": ("Hoy hace buen tiempo.", "Por favor, revise su configuración e inténtelo de nuevo."),
    "Italian": ("Oggi il tempo è bello.", "Controlla le impostazioni e riprova."),
    "Polish": ("Dziś jest ładna pogoda.", "Sprawdź swoje ustawienia i spróbuj ponownie."),
    "Portuguese": ("Hoje o tempo está bom.", "Verifique as suas configurações e tente novamente."),
    "Dutch": ("Vandaag is het mooi weer.", "Controleer uw instellingen en probeer het opnieuw."),
}
AMBIGUOUS = set("a to do me my in i o je list".split())       # English words that are Slovak, Czech or Polish words too
ENGLISH = set("the is are and how what why of to in a an for with does do explain compare write about which give me my you can "
              # (the asking words too, none of them a word of another language here - not "on" or "by": "Who was Milan
              # Rastislav Štefánik?" had no English word and was taken for Slovak, 21 Nov)
              "who was were when where did born died he she his her they this that it has have had will would from tell "
              "show list many much old at or not be".split())
LANG_SETS = {k: set(v.split()) for k, v in LANGS.items()}
SHARED = {"Slovak": "šžčťďňýáíéú", "Czech": "šžčťďňýáíéú"}   # letters of both: one point each, their own letters two
MARKS = {"Slovak": "äôľĺŕ", "Czech": "řůě", "German": "ßäöü", "Polish": "łżśćńąę", "Spanish": "ñ¿¡", "Portuguese": "ãõç", "French": "çèêàù"}


# "Same answer in Czech, please.", "now in German", "napíš to po nemecky": the answer is to be in that language. Not a
# question about a word of it ("how do you say thank you in Czech?"), which is answered in the question's language.
# (A 2-bit model wrote a Czech first line and the five tips in English, 19 Nov: with the language known, the answer is
# asked for in it and what is left in English is written again.)
LANG_NAMES = {"Slovak": r"slovak|slovensky|slovenčin\w*", "Czech": r"czech|česky|cesky|češtin\w*", "German": r"german|deutsch|nemecky|německy",
              "French": r"french|français|francúzsky|francouzsky", "Spanish": r"spanish|español|španielsky|španělsky",
              "Italian": r"italian|italiano|taliansky|italsky", "Polish": r"polish|polsku|poľsky|polsky",
              "Portuguese": r"portuguese|português|portugalsky", "Dutch": r"dutch|nederlands|holandsky"}
ABOUT_A_WORD = r"how (do|would|can) (you|i|we) (say|write|spell|pronounce)|what (is|does|'s)|what's|word for|mean\w*|ako sa povie|jak se řekne|wie sagt man"


ASKS_TO_WRITE = (r"\W*((and|now|ok|okay|please|then|also|a|teraz|ešte|teď) )*((the )?same|again|answer|reply|respond|write|say|repeat|"
                 r"translat\w*|rewrite|give|tell|explain|put|that|this|it|can you|could you|in|into|napíš|napiš|povedz|řekni|"
                 r"odpovedz|odpověz|prelož\w*|přelož\w*|to isté|totéž|po|auf|en|"
                 + "|".join(LANG_NAMES.values()) + r")\b")


def language_asked(text):
    """-> the language a message asks its answer to be written in ("Czech" for "Same answer in Czech, please."), or ""
    when it names none, names more than one, or asks about a word of it."""
    t = " ".join(str(text or "").lower().split())
    if not t or len(t.split()) > 14 or re.search(ABOUT_A_WORD, t) or re.search(r"[\"“„«]", t):
        return ""
    # (it has to open as a request to write: "I am fluent in German" and "is it written in German?" name no such wish)
    if not re.match(ASKS_TO_WRITE, t):
        return ""
    hits = [name for name, rx in LANG_NAMES.items()
            if re.search(rf"\b(in|into|po|do|auf|en|em|in het)\s+({rx})\b|\b(translat\w*|prelož\w*|přelož\w*)\b.*\b(to|do)\s+({rx})\b|"
                         rf"^\W*({rx})\W*(please|prosím)?\W*$", t)]
    return hits[0] if len(hits) == 1 else ""


def list_count(why, text):
    """-> N when a message asks for a list of N things and for no other form ("Give me 7 tips for better sleep"), else 0.
    why: the reason sizing.choose gave (a set form names the count it saw)."""
    m = re.match(r'a set form \("(\d{1,2})\s', str(why or ""))
    if not m or not 2 <= int(m.group(1)) <= 30:
        return 0
    t = str(text or "").lower()
    if re.search(r"\b(table|tabu[ľl]k\w*|tabelle|step[- ]by[- ]step|detail\w*|podrobn\w*|ausführlich\w*|essay|report|faq|plan|schedule|email|e-mail|code)\b", t):
        return 0
    return int(m.group(1))


def language_of(text):
    """-> the language of a question ("Slovak", "German", ...), or "" for English or when unsure."""
    t = re.sub(r"https?://\S+", " ", str(text or "").lower())     # (a link's words are no language)
    # distinct words ("La La Land review" is not French); not letters glued to digits: "i9-14900K" gave the Czech words
    # "i" and "k", and a search query about a processor was taken for Czech (21 Nov)
    words = set(re.findall(r"(?<![^\W_])[^\W\d_]+(?![^\W_])", re.sub(r"[^\W_]*\d[\w-]*", " ", t)))
    if not words:
        return ""
    eng = sum(w in ENGLISH for w in words)
    best, score, best_words = "", 0, 0
    for name, ws in LANG_SETS.items():
        n_words = sum(w in ws for w in words)
        # accented letters count, but only up to a point: an English question about "Štefánik" and "Na pamäť
        # Štefánikovi" scored as Slovak on its names' letters alone, and was answered in Slovak (19 Nov)
        own = sum(ch in MARKS.get(name, "") for ch in t)
        marks = min(6, 2 * own + sum(ch in SHARED.get(name, "") for ch in t))
        # (a language's own letters decide a tie: a Czech answer with five of ě/ů/ř and the words "o" and "z" scored as
        # Slovak, the shared accents alone filling the cap for both, 21 Nov)
        # (and then its words: "Quién fue Simón Bolívar?" had two Spanish words and two accents Slovak shares, and
        # Slovak, first in the list, won the tie)
        s = n_words + marks + 0.01 * own + 0.005 * n_words
        # (with none of its letters, a language needs a word of its own, not only ones English has too: "How do I add a
        # pause feature?" had "do" and "i", and a follow-up to an English chat was answered in Czech, 21 Nov)
        if not marks and not any(w in ws and w not in AMBIGUOUS for w in words):
            continue
        if s > score:
            best, score, best_words = name, s, n_words
    # English words that are words of the best language too do not count for English ("Was will ich?", "Hoe was het
    # weer?" were taken for English once "was" and "will" were English words, 21 Nov review)
    other = LANG_SETS.get(best, set()) | AMBIGUOUS
    eng = sum(w in ENGLISH and w not in other for w in words)
    if eng >= 5 and best_words * 3 <= eng:        # English sentences with a few foreign names or titles in them
        return ""
    # ... and short ones: a name's letters are no language ("Who was Milan Rastislav Štefánik?" was taken for Slovak,
    # 21 Nov). Counted without the English words that are words of other languages too ("a", "to", "do", "my", "me")
    if eng >= 2 and best_words * 2 <= eng:
        return ""
    return best if score >= 2 and score > eng else ""


def story_scenes(text, want="auto"):
    """(scenes, chapters asked) when the message asks for a story that many lanes write, a scene each; (0, 0) when one
    lane writes it: not a story (a poem, a song, a script), the next part of one, a very short one, or "Short" chosen.
    want: the length chosen in the composer."""
    t = " ".join(str(text or "").lower().split())
    if len(t.split()) > 150 or want == "short" or re.search(NOT_CREATIVE, t) or re.search(STORY_NEXT, t):
        return 0, 0
    m = re.search(CREATIVE, t) or re.search(INTL_CREATIVE, t)
    if not m or not re.match(STORY_KIND, m.group(m.lastindex or 0)) or re.search(STORY_SMALL, t):
        return 0, 0
    scenes = STORY_SCENES.get(want, 6)
    if want == "auto":
        if any(re.search(p, t) for p, _ in BIG) or re.search(r"\b(long|longer|detailed|epic|dlh\w*|dlouh\w*|lang\w*|longue?|larg[oa])\b", t):
            scenes = 12
        elif re.search(r"\b(short|krátk\w*|kurz\w*|court\w*|cort[oa])\b", t):
            scenes = 4
    wm = re.search(N + r"[- ]words?\b", t)
    if wm:
        n = int(wm.group(1).replace(",", ""))
        if n < 450:
            return 0, 0
        scenes = max(3, min(60, round(n / SCENE_WORDS)))
    cm = re.search(r"\b(\d{1,2}|two|three|four|five|six|seven|eight|nine|ten|twelve)[- ](chapters?|parts?|acts?)\b", t)
    chapters = 0
    if cm:
        chapters = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}.get(cm.group(1)) or int(cm.group(1))
        chapters = max(2, min(12, chapters))
        scenes = max(scenes, chapters * 2) if not wm else max(scenes, chapters * 2)
        scenes = min(scenes, chapters * 5)
    return scenes, chapters


# a sentence that tells about the speaker, and words that make such a sentence a request after all
TOLD = (r"(and |also |btw,? |by the way,? )?(my name is|my name's|i am|i'm|i live|i work|i was born|i come from|i study|i have|i've got|"
        r"i like|i love|i enjoy|i speak|we are|we live|volám sa|som|bývam|žijem|pracujem|mám|študujem|jmenuji se|jsem|bydlím|"
        r"pracuji|ich heiße|ich bin|ich wohne|ich arbeite|je m'appelle|je suis|j'habite|je travaille|me llamo|soy|vivo en|trabajo)\b")
TOLD_BUT_ASKS = (r"\b(need|want|wish|looking|trying|problem|issue|error|question|wonder\w*|help|would like|can't|cannot|unable|stuck|"
                 r"how|what|why|please|should|advice|recommend\w*|suggest\w*|potrebujem|chcem|pomôž\w*|pomoc|prosím|poraď\w*|"
                 r"potřebuji|chci|brauche|möchte|hilfe|bitte)\b")


# a diagram asked for in plain words (the app draws Mermaid in the answer): "create a comprehensive diagram to explain
# the rise and fall of rome, nicely colored" was planned as an article ("comprehensive"), and the model, not knowing it
# could draw, wrote "I cannot generate a colored diagram" and 1,741 words of outline, 21 Nov
DIAGRAM_WORD = (r"(diagrams?|flow ?charts?|flow diagrams?|mind ?maps?|org(?:ani[sz]ation(?:al)?)? charts?|sequence diagrams?|gantt(?: charts?)?|"
                r"infographics?|concept maps?|state (?:diagrams?|machines?)|timeline diagrams?|visual timelines?|decision trees?|"
                r"family trees?|schemas?|schémy?|schém\w*|diagram\w*|grafy?|myšlienkov\w* map\w*|myšlenkov\w* map\w*)")
DIAGRAM_ASK = (r"\b((?:create|make|draw|show|give|generate|design|build|produce|sketch|visuali[sz]e|illustrate|map out|render|"
               r"nakresli|vytvor|vytvoř|urob|udělej|zobraz|zeichne|erstelle|dessine|crée|dibuja|crea|narysuj|stwórz)\b[^.?!\n]{0,60}?\b"
               + DIAGRAM_WORD + r"|\b(?:as|in|into) an? (?:diagram|flow ?chart|mind ?map|timeline diagram)\b|\b" + DIAGRAM_WORD
               + r" (?:of|for|showing|that shows|explaining|to explain)\b)")
NOT_DIAGRAM = r"\b(what is an?|what are|explain what|how do (?:i|you) (?:read|use)|difference between)\b[^.?!]{0,30}\b(diagram|chart|map)"


def choose(text, follow_up=False):
    """follow_up: the message continues a conversation. A short one that does not ask for a text of its own ("no, you
    didn't scan it properly", "his name is Novak", "what's his name?") is answered by one lane, as a reply: the
    planner outlined such turns as reports again and again (18 Nov: three corrections, three 900-token essays with the
    same headings, none taking the correction)."""
    raw = str(text or "")
    t = " ".join(raw.lower().split())
    if len(t.split()) > 150:
        # a long message is mostly pasted material: its own cues ("brief answer", "report", "detailed") are not the
        # user's. The request is a short paragraph before or after it ("Summarize this for a busy manager:")
        paras = [" ".join(p.lower().split()) for p in re.split(r"\n\s*\n", raw) if p.strip()]
        ask = [p for p in (paras[:1] + paras[-1:] if len(paras) > 1 else []) if len(p.split()) <= 40]
        t = " … ".join(ask) if ask else " ".join(t.split()[:12]) + " … " + " ".join(t.split()[-12:])
    w = web_page(t)
    if w:
        return {"size": "short", "max_tokens": 8000, "why": f'code ("{w}") for the browser, one page by one lane', "web": True}
    dg = re.search(DIAGRAM_ASK, t)
    if dg and not re.search(NOT_DIAGRAM, t):
        return {"size": "short", "max_tokens": 3500, "why": f'a diagram ("{dg.group(0)[:40]}") is drawn by one lane', "diagram": True}
    if (re.search(CREATIVE, t) or re.search(INTL_CREATIVE, t) or re.search(STORY_NEXT, t)) and not re.search(NOT_CREATIVE, t):
        longer = any(re.search(p, t) for p, _ in BIG) or re.search(r"\b(long|longer|detailed|epic|chapters?)\b", t)
        cap = 3500 if longer else 1600
        # a length asked is the length written ("a 3000-word story" stopped at 1,600 tokens, 18 Nov): past the
        # drafting context the lane goes on as an ordinary one, up to what the conversation context holds (the
        # server lowers it to that)
        m = re.search(N + r"[- ]words?\b", t)
        if m:
            cap = max(cap, min(12000, int(int(m.group(1).replace(",", "")) * 1.45) + 150))
        return {"size": "short", "max_tokens": cap, "why": "a story or poem is written by one lane"}
    # "continue" after an answer asks for more of it: as "a short follow-up" (answer in a few sentences) it got a
    # three-sentence description of what would come next instead (19 Nov)
    if follow_up and re.fullmatch(GO_ON, t.strip()):
        return {"size": "short", "max_tokens": 1500, "why": "go on"}
    # (not a set form asked for in a follow-up: "Udělej z toho tréninkový plán na 4 týdny" was told to answer in a few
    # sentences and gave 55 words instead of a plan, 19 Nov)
    if follow_up and not any(re.search(p, t) for p, _ in BIG + [(INTL_DETAILED, ""), (INTL_REPORT, "")]) and not any(
            re.search(p, t) for p in (SEQUENCE, FORMAT, INTL_FORMAT, INTL_COUNT)) and (
            (len(t.split()) <= FOLLOW_UP_WORDS and not re.search(ASKS_FOR_TEXT, t)) or (len(t.split()) <= 40 and re.search(REFERS_BACK, t))):
        # the usual one-lane length: a reply stays short by itself, but "now add type hints, error handling and unit
        # tests" is code, and at 600 tokens it was cut off in the middle (18 Nov)
        return {"size": "short", "max_tokens": 1500, "why": "a short follow-up"}
    m = re.search(r"\b(?:in|about|around|roughly|approximately|of|max(?:imum)?|at most|under|up to|no more than)\s+" + N + r"\s*(?:-|to)?\s*(?:\d[\d,]*)?\s*words?\b", t) \
        or re.search(N + r"[- ]words?\b", t)
    per_part = None
    if m and re.search(r"\b(each|every|per)\b", t[max(0, m.start() - 40):m.end() + 20]):
        # "keep each section under 50 words": a length per part, not for the whole answer (18 Nov); it is given to the
        # plan as the length of a part (with the size the rest of the message asks for)
        per_part, m = int(m.group(1).replace(",", "")), None
    # "in two paragraphs", "three paragraphs": that many paragraphs, by one lane (planned, two paragraphs became a
    # report of sections and bullets, 18 Nov)
    # ("a five-paragraph essay" too: planned, it came out as 12 paragraphs, 19 Nov)
    pm = re.search(r"\b(two|three|four|five|six|seven|2|3|4|5|6|7)[\s-]+(short\s+|long\s+)?paragraphs?\b", t)
    if pm and not m:
        k = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}.get(pm.group(1), None) or int(pm.group(1))
        return {"size": "short", "max_tokens": 200 * k + 100, "why": f"{k} paragraphs asked"}
    if m:
        n = int(m.group(1).replace(",", ""))
        if n <= 150:
            return {"size": "short", "max_tokens": int(n * 1.8) + 40, "why": f"{n} words asked"}
        if n <= 700:
            return {"size": "answer", "max_tokens": 1500, "why": f"{n} words asked", "words": n}
        if n <= 1800:
            return {"size": "detailed", "max_tokens": 1500, "why": f"{n} words asked", "words": n}
        if n <= 5000:
            return {"size": "report", "max_tokens": 1500, "why": f"{n} words asked", "words": n}
        return {"size": "long", "max_tokens": 1500, "why": f"{n} words asked", "words": n}
    for pat, cap, why in BRIEF + [(INTL_BRIEF, 300, "brevity asked")]:
        bm = re.search(pat, t)
        # ("summarise each chapter in one line": a length per item too)
        if bm and not (cap < 300 and re.search(r"\b(each|every|per)\b", t[max(0, bm.start() - 30):bm.start()])):
            return {"size": "short", "max_tokens": cap, "why": why}
    hit = re.search(CODE, t) or (re.search(TEACH, t) and re.search(TECH, t))
    if hit and not re.search(r"\b(no|without) (code|commands)\b", t):
        # code asked for is written by one lane, in code blocks: planned bullets cannot hold them ("a detailed tutorial
        # on Python decorators with code examples" came back as 1,000 words of bullets without a line of code, 18 Nov).
        # So is an explanation of a programming subject, which a single lane shows with code ("Explain how Python
        # decorators work": planned, one code block)
        big = any(re.search(p, t) for p, _ in BIG) or re.search(INTL_DETAILED, t) or re.search(INTL_REPORT, t)
        # (4,000: a whole program is worth a few more seconds; cut at 2,500 it does not run, 21 Nov)
        return {"size": "short", "max_tokens": 6000 if big else 4000, "why": f'code ("{hit.group(0)}") is written by one lane'}
    for pat, size in BIG[:2] + [(INTL_REPORT, "report")]:   # a book or a report is planned even with tables in it
        hit = re.search(pat, t)
        if hit:
            return {"size": size, "max_tokens": 1500, "why": f'"{hit.group(0)}"', "per_part": per_part}
    for pat, why in ((PERSONA, "a role or a style is kept by one lane"), (SIMPLE, "said simply, for a young reader, by one lane")):
        hit = re.search(pat, t)
        if hit and pat is SIMPLE and (re.search(FORMAT, t) or re.search(INTL_COUNT, t)):
            hit = None                        # a list of a set length for children: its count decides the room (below)
        if hit:
            big = any(re.search(p, t) for p, _ in BIG) or re.search(INTL_DETAILED, t)
            return {"size": "short", "max_tokens": 4000 if big and pat is PERSONA else 1500 if pat is PERSONA else 700,
                    "why": f'{why} ("{hit.group(0)}")'}
    hit = re.search(SEQUENCE, t)
    if hit:
        big = any(re.search(p, t) for p, _ in BIG) or re.search(INTL_DETAILED, t)
        return {"size": "short", "max_tokens": 4000 if big else 2500, "why": f'times or sums that must agree ("{hit.group(0)}") are written by one lane'}
    hit = re.search(AVOID_WORD, t)
    if hit:
        big = any(re.search(p, t) for p, _ in BIG) or re.search(INTL_DETAILED, t)
        return {"size": "short", "max_tokens": 4000 if big else 1500, "why": f'a word to leave out ("{hit.group(0)}") is kept to by one lane'}
    hit = re.search(FORMAT, t) or re.search(INTL_FORMAT, t) or re.search(INTL_COUNT, t)
    if hit:
        # (a detailed one gets the room for it: "a detailed FAQ" stopped at 1,500 tokens in its ninth answer, 18 Nov)
        big = any(re.search(p, t) for p, _ in BIG) or re.search(INTL_DETAILED, t)
        cap = 4000 if big else 1500
        # as many items as asked: "translate this to 50 languages" needs room for 50 lines
        counts = [int(x) for x in re.findall(r"\b(\d{2,3})\b(?=\s+(?:\w+\s+)?(?:languages|items|ideas|names|examples|questions|words|facts|tips|steps|lines|entries))", t)]
        if counts:
            cap = max(cap, min(8000, max(counts) * 70 + 300))
        return {"size": "short", "max_tokens": cap, "why": f'a set form ("{hit.group(0)}") is written by one lane'}
    for pat, size in BIG[2:] + [(INTL_DETAILED, "detailed")]:
        hit = re.search(pat, t)
        if hit:
            return {"size": size, "max_tokens": 1500, "why": f'"{hit.group(0)}"', "per_part": per_part}
    # a short question ("Define entropy.", "tabs or spaces?", "can dogs eat grapes?") is answered by one lane, as a reply:
    # the model's router planned them into 500-760-word reports with sections (18 Nov)
    words = t.split()
    # a greeting or small talk is a reply, whatever the model's router says (Nemotron planned "Ako sa máš? ... povedz
    # mi, čo vieš robiť" into five lanes, 18 Nov)
    # a request that needs a conversation before it, in a new chat ("tell me more", "continue", "pokračuj"): a short
    # question back, not a menu of topics (one came out at 207 words of options, 19 Nov)
    if not follow_up and re.fullmatch(CONTEXT_FREE, t.strip()):
        return {"size": "short", "max_tokens": 200, "why": "nothing to go on yet"}
    gm = re.match(r"(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|ahoj|čau|cau|dobrý deň|dobry den|"
                  r"ako sa máš|ako sa mas|jak se máš|hallo|guten (tag|morgen|abend)|wie geht|bonjour|salut|hola|"
                  r"buenos días|ciao|buongiorno|cześć|dzień dobry|olá|hoi|goedemorgen)\b", t)
    # (only when what follows is small talk too: "Ahoj, plánujem dovolenku v Chorvátsku s deťmi. Čo odporúčaš?" is a
    # question with a greeting in front; as a greeting it got 1,500 tokens of one lane and was cut off, 19 Nov)
    if len(words) <= 20 and gm and len(t[gm.end():].split()) <= 6:
        return {"size": "short", "max_tokens": 1500, "why": "a greeting"}
    # something told about oneself, with nothing asked ("My name is Petra and I live in Brno. I work as a nurse."): a
    # reply. A model's router planned it into five headed sections, 612 words (Qwen3.8-27B HauhauCS IQ4_XS, 19 Nov)
    if not follow_up and len(words) <= 40 and "?" not in t and not re.search(TOLD_BUT_ASKS, t) and not re.search(ASKS_FOR_TEXT, t):
        said = [x.strip() for x in re.split(r"[.!;]+\s*", t) if x.strip()]
        if said and all(re.match(TOLD, x) for x in said):
            return {"size": "short", "max_tokens": 600, "why": "something told, not asked"}
    asks = t.rstrip().endswith("?") or re.match(r"(define|what|who|when|where|which|why|how (much|many|long|old|far|big))\b", t)
    # (one word fewer in a language without articles: "Aké boli hlavné príčiny Francúzskej revolúcie?" is 6 words where
    # "What were the main causes of the French Revolution?" is 9, and got a short reply where the English got a full
    # answer, 19 Nov)
    most = 6 if not language_of(t) else 5
    if asks and not re.search(ASKS_FOR_TEXT, t) and (len(words) <= most or (len(words) <= 12 and re.match(YES_NO, t))):
        return {"size": "short", "max_tokens": 1000, "why": "a short question"}
    return {"size": "auto", "max_tokens": 1500, "why": "the model decides"}


if __name__ == "__main__":
    for q in ["Write a comprehensive guide to the history, science, economics and future of solar power in one sentence.",
              "Write a comprehensive guide to the history, science, economics and future of solar power.",
              "What is 17 times 23?", "Explain TCP vs UDP in about 300 words.", "Write a 3,000-word report on inflation.",
              "Give me a practical guide to improving sleep.", "In two sentences, why is the sky blue?", "Write an exhaustive book-length overview of Rome."]:
        print(choose(q), "|", q[:70])
