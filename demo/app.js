// The page: what the visitor has picked and hidden, and drawing it. The model's
// arithmetic is in recommend.js and the choosing of what to show in select.js.
import { scoreAll, topGames, unitVectors, userVector } from './recommend.js'
import { byRatings, closestPick, hiddenGem, search } from './select.js'

// set when the repository is public
const CODE_URL = ''
const STARTERS = {
    'Modern classics': ['Catan', 'Ticket to Ride', 'Carcassonne', 'Pandemic', '7 Wonders', 'Splendor'],
    'Childhood favourites': ['Monopoly', 'Scrabble', 'Clue', 'Risk', 'Battleship', 'UNO'],
    'Party night': ['Codenames', 'Dixit', 'Just One', 'Wavelength', 'Telestrations', 'The Resistance: Avalon'],
    'Family table': ['Azul', 'Kingdomino', 'Sushi Go!', 'King of Tokyo', 'Qwirkle', 'Blokus'],
    'Heavy strategy': ['Terraforming Mars', 'Scythe', 'Brass: Birmingham', 'Gloomhaven', 'Terra Mystica', 'Agricola'],
}
const WELL_KNOWN = 100 // what the switch hides
const FAMILIAR = 300 // games a visitor has plausibly played, offered as further picks

const $ = (id) => document.getElementById(id)

/** el('button', { class: 'chip', onclick }, 'text', child) */
function el(tag, props = {}, ...children) {
    const node = Object.assign(document.createElement(tag), props)
    if (props.class) node.className = props.class
    node.append(...children.filter((child) => child != null))
    return node
}

const icon = (name) => el('i', {}, name)

async function load() {
    const [saved, buffer] = await Promise.all([
        fetch('data/model.json').then((response) => response.json()),
        fetch('data/game_vectors.bin').then((response) => response.arrayBuffer()),
    ])
    const model = { d: saved.embedding_size, meanRating: saved.mean_rating, tower: saved.tower, vectors: new Float32Array(buffer) }
    const games = saved.games
    const order = byRatings(games.ratings)
    const rank = new Uint32Array(order.length)
    order.forEach((game, position) => (rank[game] = position))
    return {
        model, games, order, rank,
        unit: unitVectors(model.vectors, model.d),
        lowerNames: games.name.map((name) => name.toLowerCase()),
    }
}

const state = { picks: [], hidden: [], query: '', hideKnown: false }
let data

// The lists live in the address as BoardGameGeek ids, so a reload or a shared link keeps them.
function remember() {
    const ids = (list) => list.map((game) => data.games.id[game]).join(',')
    history.replaceState(null, '', `#picks=${ids(state.picks)}&hidden=${ids(state.hidden)}`)
}
function recall() {
    const saved = new URLSearchParams(location.hash.slice(1))
    const games = (key) => (saved.get(key) || '').split(',').map((id) => data.games.id.indexOf(Number(id))).filter((game) => game >= 0)
    state.picks = games('picks')
    state.hidden = games('hidden')
}

function update(change) {
    change()
    remember()
    draw()
}
const pick = (game) => update(() => {
    state.hidden = state.hidden.filter((other) => other !== game)
    if (!state.picks.includes(game)) state.picks.push(game)
    state.query = ''
    $('search').value = ''
})
// a pick that is removed is still a game the visitor knows, so it is hidden, not offered again
const hide = (game) => update(() => {
    state.picks = state.picks.filter((other) => other !== game)
    if (!state.hidden.includes(game)) state.hidden.push(game)
})
const restore = (game) => update(() => (state.hidden = state.hidden.filter((other) => other !== game)))

function title(game) {
    const year = data.games.year[game]
    return year > 0 ? `${data.games.name[game]} (${year})` : data.games.name[game]
}

function chip(game, onclick, iconName) {
    return el('button', { class: 'chip', onclick: () => onclick(game) }, iconName ? icon(iconName) : null, el('span', {}, title(game)))
}

function recommendation(game) {
    const because = closestPick(data.unit, data.model.d, game, state.picks)
    return el('li', {},
        el('div', { class: 'max' },
            el('a', { class: 'link', href: `https://boardgamegeek.com/boardgame/${data.games.id[game]}`, target: '_blank', rel: 'noreferrer' },
                el('h6', { class: 'small' }, title(game))),
            el('div', { class: 'small-text' }, `Closest to your pick ${data.games.name[because]}`)),
        el('button', { class: 'border small', onclick: () => pick(game) }, icon('thumb_up'), el('span', {}, 'Played it, liked it')),
        el('button', { class: 'border small', onclick: () => hide(game) }, icon('visibility_off'), el('span', {}, 'Not for me')))
}

function draw() {
    const { games, order, rank, unit, model, lowerNames } = data
    const taken = new Set([...state.picks, ...state.hidden])

    let recommended = []
    let ranked = []
    if (state.picks.length) {
        const scores = scoreAll(unit, model.d, userVector(model, state.picks))
        ranked = topGames(scores, taken, scores.length)
        recommended = (state.hideKnown ? ranked.filter((game) => rank[game] >= WELL_KNOWN) : ranked).slice(0, 10)
    }

    // under the search box: matches while typing, otherwise more games to pick
    let offered
    if (state.query.trim()) {
        offered = search(lowerNames, games.ratings, state.query, taken, 8)
        $('suggestions-label').textContent = offered.length ? '' : 'No game by that name.'
    } else if (state.picks.length) {
        const shown = new Set(recommended)
        offered = ranked.filter((game) => rank[game] < FAMILIAR && !shown.has(game)).slice(0, 8)
        $('suggestions-label').textContent = 'Similar to your picks, that you might have also played:'
    } else {
        offered = order.slice(0, 16)
        $('suggestions-label').textContent = 'Or tap a well-known game:'
    }
    $('suggestions').replaceChildren(...offered.map((game) => chip(game, pick, 'add')))

    const n = state.picks.length
    $('picks-label').textContent = n ? `Your picks (${n})` : 'Your picks'
    $('picks').replaceChildren(...(n
        ? state.picks.map((game) => chip(game, hide, 'close'))
        : [el('span', { class: 'small-text' }, 'Nothing yet. Tap a set above, or search for a game.')]))
    $('hidden-box').hidden = !state.hidden.length
    $('hidden-label').textContent = `Hidden from recommendations (${state.hidden.length})`
    $('hidden').replaceChildren(...state.hidden.map((game) => chip(game, restore, 'undo')))

    $('results').hidden = !n
    $('status').textContent = !n ? '' : n < 10
        ? `${n} pick${n === 1 ? '' : 's'} so far. The recommendations sharpen as you add more; ten is a good number.`
        : ''
    if (!n) return
    $('recommendations').replaceChildren(...recommended.map(recommendation))

    const gem = hiddenGem(ranked, games, rank, state.picks, new Set([...taken, ...recommended]))
    $('gem').hidden = gem === undefined
    if (gem !== undefined) {
        $('gem').replaceChildren(
            el('nav', {},
                icon('diamond'),
                el('div', { class: 'max' },
                    el('div', { class: 'small-text' }, 'Hidden gem: your best match outside the 1,000 best-known games'),
                    el('a', { class: 'link', href: `https://boardgamegeek.com/boardgame/${games.id[gem]}`, target: '_blank', rel: 'noreferrer' },
                        el('h6', { class: 'small' }, title(gem))),
                    el('div', { class: 'small-text' },
                        `Rated ${games.average[gem].toFixed(1)} by ${games.ratings[gem].toLocaleString()} people. Closest to your pick ${games.name[closestPick(unit, model.d, gem, state.picks)]}`)),
                el('button', { class: 'border small', onclick: () => hide(gem) }, icon('visibility_off'), el('span', {}, 'Not for me'))))
    }
}

function socialLinks() {
    // the portfolio's own icons; "mode" is shared with it when both are on the same site
    let mode = 'light'
    try {
        mode = globalThis.ui?.('mode', localStorage.getItem('mode') || 'auto') ?? mode
    } catch {
        // storage can be blocked; the page works without it
    }
    if (mode !== 'dark' && mode !== 'light') mode = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    const links = { LinkedIn: 'https://www.linkedin.com/in/billy-york/', GitHub: 'https://github.com/yorker78' }
    $('social').replaceChildren(...Object.entries(links).map(([name, href]) =>
        el('a', { href, target: '_blank', rel: 'noreferrer', ariaLabel: name, class: 'small-margin' },
            el('img', { src: `https://www.billyyork.com/assets/img/${name.toLowerCase()}-${mode}.png`, alt: name }))))
}

async function start() {
    socialLinks()
    if (CODE_URL) Object.assign($('code-link'), { href: CODE_URL, hidden: false })
    try {
        data = await load()
    } catch (error) {
        $('status').textContent = 'The model could not be loaded. Please try again later.'
        throw error
    }
    // a starter set fills in its games, most-rated edition of each name
    $('starters').replaceChildren(...Object.entries(STARTERS).map(([label, names]) => {
        const set = names.map((name) => data.order.find((game) => data.games.name[game] === name)).filter((game) => game !== undefined)
        return el('button', { class: 'chip fill', onclick: () => update(() => {
            state.picks = set.slice()
            state.hidden = state.hidden.filter((game) => !set.includes(game))
        }) }, el('span', {}, label))
    }))
    $('search').addEventListener('input', (event) => update(() => (state.query = event.target.value)))
    $('hide-known').addEventListener('change', (event) => update(() => (state.hideKnown = event.target.checked)))
    recall()
    draw()
}

// BeerCSS's script sets `ui` when it has loaded; modules run in order, so it is ready here
start()
