import assert from 'node:assert/strict'
import { test } from 'node:test'

import { byRatings, closestPick, hiddenGem, isCrude, search } from './select.js'

const games = {
    name: ['Catan', 'Catan Big Box', 'Obscure Gem', 'Bad Obscure', 'Catacombs', 'Other Gem'],
    average: [7.1, 7.0, 8.0, 5.0, 7.2, 7.5],
}
const ratings = [900, 300, 20, 10, 100, 15]
const lower = games.name.map((n) => n.toLowerCase())

test('byRatings orders games from most to least rated', () => {
    assert.deepEqual(byRatings(ratings), [0, 1, 4, 2, 5, 3])
})

test('search puts names that start with the query first, then the best known, and skips excluded games', () => {
    assert.deepEqual(search(lower, ratings, 'cat', new Set(), 5), [0, 1, 4])
    assert.deepEqual(search(lower, ratings, 'gem', new Set([2]), 5), [5])
    assert.deepEqual(search(lower, ratings, '  ', new Set(), 5), [])
})

test('isCrude catches crude titles and leaves ordinary ones, including near misses', () => {
    for (const name of ['Eat Poop You Cat', 'Shit Happens', 'Smart Ass', 'Exploding Kittens: NSFW Deck', 'F**k. The game']) assert.ok(isCrude(name), name)
    for (const name of ['Assassin', 'Battle of the Sexes', 'Craps', 'Button Men', 'Secret Hitler', 'Oh Hell!', 'Seaweed']) assert.ok(!isCrude(name), name)
})

test('closestPick finds the pick pointing the same way as the game', () => {
    const unit = new Float32Array([1, 0, 0, 1, 0.6, 0.8])
    assert.equal(closestPick(unit, 2, 2, [0, 1]), 1)
})

test('hiddenGem skips well-known, poorly rated, hidden and same-series games', () => {
    const order = [1, 0, 3, 2, 5, 4] // best match first
    const rank = [0, 1, 3, 5, 2, 4] // position in byRatings
    const options = { beyond: 1, average: 7 }
    // 1 is another Catan, 3 is rated 5.0, so the first that qualifies is 2
    assert.equal(hiddenGem(order, games, rank, [0], new Set([0]), options), 2)
    assert.equal(hiddenGem(order, games, rank, [0], new Set([0, 2]), options), 5)
    assert.equal(hiddenGem(order, games, rank, [0], new Set([0, 2, 5, 4]), options), undefined)
})
