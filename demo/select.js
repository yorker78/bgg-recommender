// Choosing what the page shows from a visitor's scores. No page code here, so
// it can be checked from the command line: node --test demo/select.test.js

/** Game indices from the most-rated game to the least. */
export function byRatings(ratings) {
    return Array.from(ratings.keys()).sort((a, b) => ratings[b] - ratings[a])
}

/** Games whose name contains the query: names that start with it first, then the best known. */
export function search(lowerNames, ratings, query, excluded, count) {
    const q = query.trim().toLowerCase()
    if (!q) return []
    const starts = (i) => (lowerNames[i].startsWith(q) ? 1 : 0)
    return Array.from(lowerNames.keys())
        .filter((i) => lowerNames[i].includes(q) && !excluded.has(i))
        .sort((a, b) => starts(b) - starts(a) || ratings[b] - ratings[a])
        .slice(0, count)
}

/**
 * Titles the page will not suggest on its own, since it is shown to people who
 * did not come for crude humour. A visitor can still search for them by name.
 * ponytail: a short word list, not a rating of the game; extend it when one slips through
 */
const CRUDE = /\b(poop|poo|shit|f\*+k|fuck\w*|sex|orgas\w*|porn\w*|nsfw|ass|crap|crappy|fart|stoner|stoned|drunk|weed)\b/i
export const isCrude = (name) => CRUDE.test(name)

/** The pick most like a game: the "because you picked" line. */
export function closestPick(unit, d, game, picks) {
    let best = picks[0]
    let bestScore = -Infinity
    for (const pick of picks) {
        let score = 0
        for (let j = 0; j < d; j++) score += unit[game * d + j] * unit[pick * d + j]
        if (score > bestScore) [best, bestScore] = [pick, score]
    }
    return best
}

// "Ticket to Ride: Europe" and "Carcassonne Big Box 6" are more of a game the visitor already picked
const series = (name) => name.split(/[:(]/)[0].trim().toLowerCase()

/**
 * The best match that is not well known: outside the `beyond` most-rated games,
 * rated `average` or more by the community, and not an edition or spin-off of
 * a pick or of a well-known game ("Ticket to Ride: Europe - 15th Anniversary").
 * `byRating` is every game from most to least rated. Returns undefined when
 * nothing qualifies.
 */
export function hiddenGem(order, games, byRating, rank, picks, excluded, { beyond = 1000, average = 7 } = {}) {
    const familiar = [...byRating.slice(0, beyond), ...picks]
    const names = new Set(familiar.map((i) => series(games.name[i])))
    const families = new Set(familiar.map((i) => games.family?.[i]).filter(Boolean))
    // ponytail: matching on the start of the name; the family column, once exported, catches translated titles
    const starts = [...names].filter((name) => name.length >= 5).map((name) => name + ' ')
    const spinOff = (i) => {
        const name = games.name[i].toLowerCase()
        return names.has(series(name)) || starts.some((start) => name.startsWith(start)) || families.has(games.family?.[i])
    }
    return order.find((i) => rank[i] >= beyond && games.average[i] >= average && !excluded.has(i) && !spinOff(i))
}

const CATEGORY_NAMES = { Strategy: 'strategy', Thematic: 'thematic', War: 'war', Family: 'family', CGS: 'card', Abstract: 'abstract', Party: 'party', Childrens: "children's" }

/** One sentence on what the picks have in common, from whatever the export carries; '' if it carries none of it. */
export function glance(games, picks) {
    const mean = (column) => {
        const values = picks.map((i) => column?.[i]).filter((value) => value > 0)
        return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0
    }
    const parts = []
    const counts = Object.keys(CATEGORY_NAMES)
        .map((name) => [name, picks.filter((i) => games.categories?.[name]?.[i]).length])
        .filter(([, count]) => count > 0)
        .sort((a, b) => b[1] - a[1])
    if (counts.length) parts.push(`mostly ${counts.slice(0, 2).map(([name]) => CATEGORY_NAMES[name]).join(' and ')} games`)
    const weight = mean(games.weight)
    if (weight) parts.push(`${weight < 2 ? 'light' : weight < 3 ? 'medium' : 'heavy'} complexity (${weight.toFixed(1)} of 5)`)
    const minutes = mean(games.minutes)
    if (minutes) parts.push(`around ${Math.round(minutes / 5) * 5} minutes a game`)
    return parts.length ? `Your picks at a glance: ${parts.join(', ')}.` : ''
}
