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
 * rated `average` or more by the community, and not another edition of a pick.
 * Returns undefined when nothing qualifies.
 */
export function hiddenGem(order, games, rank, picks, excluded, { beyond = 1000, average = 7 } = {}) {
    const picked = picks.map((i) => series(games.name[i]))
    const families = new Set(picks.map((i) => games.family?.[i]).filter(Boolean))
    return order.find(
        (i) =>
            rank[i] >= beyond &&
            games.average[i] >= average &&
            !excluded.has(i) &&
            !picked.some((name) => games.name[i].toLowerCase().startsWith(name)) &&
            !families.has(games.family?.[i])
    )
}
