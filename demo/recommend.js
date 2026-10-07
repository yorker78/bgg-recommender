
export function unitVectors(vectors, d) {
    const units = new Float32Array(vectors.length)
    for (let i = 0; i < vectors.length; i+= d) {
        const vector = vectors.subarray(i, i + d)
        const len = Math.sqrt(vector.reduce((acc, cur) => acc + (cur ** 2), 0)) || 1
        units.set(vector.map((val) => val / len), i)
    }
    return units
}

export function layer(weights, bias, inputs) {
    const out = []
    for (let i = 0; i < weights.length; i++) {
        let sum = bias[i]
        for (let j = 0; j < weights[i].length; j++) {
            sum += weights[i][j] * inputs[j]
        }
        out[i] = sum
    }
    return out
}

export function userVector(model, picks) {
    const d = model.d
    const average = new Array(d).fill(0)
    for (const pick of picks) {
        const vector = model.vectors.subarray(pick * d, pick * d + d)
        for (let j = 0; j < d; j++) {
            average[j] += vector[j] / picks.length
        }
    }
    const inputs = []
    for (let i = 0; i < d; i++){
        inputs[i] = average[i] * (9 - model.meanRating)
    }
    inputs.push(...average)
    inputs.push(Math.log1p(picks.length))
    const hidden = layer(model.tower.w1, model.tower.b1, inputs).map((val) => val < 0 ? 0 : val)
    const outputs = layer(model.tower.w2, model.tower.b2, hidden)
    const len = Math.sqrt(outputs.reduce((acc, cur) => acc + (cur ** 2), 0)) || 1
    return outputs.map((val) => val / len)
}

export function scoreAll(unit, d, ideal) {
    const scores = new Float32Array(unit.length / d)
    for (let i = 0; i < scores.length; i++) {
        let sum = 0
        for (let j = 0; j < ideal.length; j++) {
            sum += unit[i*d + j] * ideal[j]
        }
        scores[i] = sum
    }
    return scores
}

export function topGames(scores, excluded, count) {
    return Array.from(scores.keys())
                .filter((i) => !excluded.has(i))
                .sort((a, b) => scores[b] - scores[a])
                .slice(0, count)
}