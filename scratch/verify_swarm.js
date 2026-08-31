export const meta = {
  name: 'verify-erasure-swarm',
  description: "Vérif visuelle paranoïaque de l'effacement (crops du barème) par essaim Sonnet",
  phases: [{ title: 'Verify', detail: 'agents Sonnet lisent les crops erased/before' }],
}

const NCH = (args && args.nchunks) || 12
const DIR = 'scratch/wl_chunks'
log(`${NCH} agents Sonnet, un par chunk`)

const SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          series: { type: 'string' },
          page: { type: 'string' },
          index: { type: 'integer' },
          verdict: { type: 'string', enum: ['PASS', 'GHOST', 'DECOR_DAMAGE', 'SPILL', 'OTHER'] },
          severity: { type: 'string', enum: ['none', 'minor', 'major'] },
          note: { type: 'string' },
        },
        required: ['series', 'page', 'index', 'verdict', 'severity', 'note'],
      },
    },
  },
  required: ['verdicts'],
}

function prompt(ci) {
  const f = `${DIR}/chunk_${String(ci).padStart(2, '0')}.json`
  return `Tu es vérificateur QUALITÉ d'un pipeline de traduction de manhwa. Mission : juger l'EFFACEMENT du texte source (avant réinjection de la traduction).

1. Lis d'abord le fichier ${f} (utilise: cat "${f}"). C'est un tableau JSON de zones ; chaque zone a: series, page, index, cls, text (texte source), et des chemins ABSOLUS "before", "erased", "after".
2. Pour CHAQUE zone, utilise l'outil Read sur le chemin "erased" (et sur "before" comme référence). N'émets JAMAIS de verdict sans avoir ouvert l'image "erased".
   - "before" = image d'origine, texte anglais source visible.
   - "erased" = APRÈS effacement, AVANT que la traduction ne soit dessinée : elle doit ne contenir AUCUN texte.

Verdict par zone :
- PASS  : tout le texte source a disparu ET le décor/dessin autour est intact.
- GHOST : reste un fantôme du texte source (même très pâle : contour, halo gris, lettres résiduelles). Sois PARANOÏAQUE.
- DECOR_DAMAGE : l'effacement a abîmé le dessin (tache blanche/noire, structure détruite, décor gommé, aplat déversé). LE PIRE défaut.
- SPILL : peinture d'effacement qui bave nettement sur du dessin propre hors du texte.
- OTHER : autre anomalie.

severity : none (PASS) | minor (à peine visible) | major (flagrant, casse la lecture).
note : une phrase précise (où + quoi), ex "fantôme pâle de GOLDEN AGE à droite" ou "RAS, arche intacte".

Renvoie EXACTEMENT un verdict par zone du chunk, via la sortie structurée.`
}

const res = await parallel(
  Array.from({ length: NCH }, (_, ci) => () =>
    agent(prompt(ci), { label: `verify:c${ci}`, phase: 'Verify', schema: SCHEMA, agentType: 'general-purpose', model: 'sonnet' })
  )
)

const all = res.filter(Boolean).flatMap(r => (r && r.verdicts) || [])
const fails = all.filter(v => v.verdict !== 'PASS')
const majors = fails.filter(v => v.severity === 'major')
const damage = fails.filter(v => v.verdict === 'DECOR_DAMAGE')
log(`verdicts=${all.length} fails=${fails.length} majors=${majors.length} decor_damage=${damage.length}`)
return {
  total: all.length,
  pass: all.length - fails.length,
  by_verdict: all.reduce((m, v) => (m[v.verdict] = (m[v.verdict] || 0) + 1, m), {}),
  decor_damage: damage,
  majors,
  fails,
}
