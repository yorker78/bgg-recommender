// Checks recommend.js against what PyTorch computed for a small made-up model.
// Run with: node --test demo/recommend.test.js
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'

import { layer, scoreAll, topGames, unitVectors, userVector } from './recommend.js'

const fixture = JSON.parse(readFileSync(new URL('./fixture.json', import.meta.url)))
const model = { ...fixture, vectors: new Float32Array(fixture.vectors) }
const close = (actual, expected) =>
    assert.ok(actual.length === expected.length && expected.every((value, i) => Math.abs(actual[i] - value) < 1e-4), `${Array.from(actual)} is not ${expected}`)

test('unitVectors gives every game a vector of length 1', () => {
    const unit = unitVectors(model.vectors, model.d)
    for (let i = 0; i < unit.length; i += model.d) {
        close([Math.hypot(...unit.subarray(i, i + model.d))], [1])
    }
})

test('layer multiplies each row of weights by the inputs and adds the bias', () => {
    close(layer([[1, 2], [0, -1], [3, 0]], [10, 20, 30], [5, 7]), [29, 13, 45])
})

test('userVector matches the ideal game PyTorch makes from three picks', () => {
    close(userVector(model, fixture.picks), fixture.ideal)
})

test('scoreAll and topGames match the scores and ranking PyTorch gives', () => {
    const scores = scoreAll(unitVectors(model.vectors, model.d), model.d, userVector(model, fixture.picks))
    close(scores, fixture.scores)
    assert.deepEqual(Array.from(topGames(scores, new Set(fixture.picks), 5)), fixture.top5)
})
