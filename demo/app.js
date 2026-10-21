// The page: what the visitor has picked and hidden, and drawing it. The model's
// arithmetic is in recommend.js and the choosing of what to show in select.js.
import { scoreAll, topGames, unitVectors, userVector } from './recommend.js'
import { byRatings, closestPick, glance, hiddenGem, isCrude, search } from './select.js'

const CODE_URL = 'https://github.com/yorker78/bgg-recommender'
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
        crude: order.filter((game) => isCrude(games.name[game])),
    }
}

const state = { picks: [], hidden: [], saved: [], query: '', hideKnown: false }
let data

// The lists live in the address as BoardGameGeek ids, so a reload or a shared link keeps them.
function remember() {
    const ids = (list) => list.map((game) => data.games.id[game]).join(',')
    history.replaceState(null, '', `#picks=${ids(state.picks)}&hidden=${ids(state.hidden)}&saved=${ids(state.saved)}`)
}
function recall() {
    const saved = new URLSearchParams(location.hash.slice(1))
    const games = (key) => (saved.get(key) || '').split(',').map((id) => data.games.id.indexOf(Number(id))).filter((game) => game >= 0)
    state.picks = games('picks')
    state.hidden = games('hidden')
    state.saved = games('saved')
}

function update(change) {
    change()
    remember()
    draw()
}
const without = (list, game) => list.filter((other) => other !== game)
const pick = (game) => update(() => {
    state.hidden = without(state.hidden, game)
    state.saved = without(state.saved, game)
    if (!state.picks.includes(game)) state.picks.push(game)
    state.query = ''
    $('search').value = ''
})
// a pick that is removed is still a game the visitor knows, so it is hidden, not offered again
const hide = (game) => update(() => {
    state.picks = state.picks.filter((other) => other !== game)
    if (!state.hidden.includes(game)) state.hidden.push(game)
})
const restore = (game) => update(() => (state.hidden = without(state.hidden, game)))
// "I have heard of this and think I would like it": kept as a shortlist. The model has
// no such signal to learn from, so it does not change the scores; it only clears the slot.
const save = (game) => update(() => state.saved.push(game))
const unsave = (game) => update(() => (state.saved = without(state.saved, game)))

function title(game) {
    const year = data.games.year[game]
    return year > 0 ? `${data.games.name[game]} (${year})` : data.games.name[game]
}

function chip(game, onclick, iconName) {
    return el('button', { class: 'chip', onclick: () => onclick(game) }, iconName ? icon(iconName) : null, el('span', {}, title(game)))
}

// BoardGameGeek's own categories, each with an icon to stand in for box art
const CATEGORY_ICONS = {
    Strategy: 'chess', Thematic: 'theater_comedy', War: 'swords', Family: 'family_restroom',
    CGS: 'playing_cards', Abstract: 'category', Party: 'celebration', Childrens: 'toys',
}
function categoryIcon(game) {
    const flags = data.games.categories ?? {}
    const category = Object.keys(CATEGORY_ICONS).find((name) => flags[name]?.[game])
    return el('i', { class: 'secondary-text', title: category ?? '' }, CATEGORY_ICONS[category] ?? 'casino')
}

/** "2-4 players, 60 min, complexity 2.3 of 5", from whatever the export carries. */
function details(game) {
    const { min_players: low, max_players: high, minutes, weight } = data.games
    const parts = []
    if (low?.[game] > 0) parts.push(high[game] > low[game] ? `${low[game]}-${high[game]} players` : `${low[game]} player${low[game] === 1 ? '' : 's'}`)
    if (minutes?.[game] > 0) parts.push(`${minutes[game]} min`)
    if (weight?.[game] > 0) parts.push(`complexity ${weight[game].toFixed(1)} of 5`)
    return parts.join(', ')
}

function recommendation(game) {
    const because = closestPick(data.unit, data.model.d, game, state.picks)
    return el('li', {},
        categoryIcon(game),
        el('div', { class: 'max' },
            el('a', { class: 'link', href: `https://boardgamegeek.com/boardgame/${data.games.id[game]}`, target: '_blank', rel: 'noreferrer' },
                el('h6', { class: 'small' }, title(game))),
            el('div', { class: 'small-text' }, details(game)),
            el('div', { class: 'small-text' }, `Closest to your pick ${data.games.name[because]}`)),
        el('button', { class: 'border small', title: 'I have played this and liked it: add it to my picks', onclick: () => pick(game) }, icon('thumb_up'), el('span', {}, 'Liked it')),
        el('button', { class: 'border small', title: 'I think I would like this: add it to my shortlist', onclick: () => save(game) }, icon('bookmark_add'), el('span', {}, 'Want to try')),
        el('button', { class: 'border small', title: 'Hide this game', onclick: () => hide(game) }, icon('visibility_off'), el('span', {}, 'Not for me')))
}

function draw() {
    const { games, order, rank, unit, model, lowerNames, crude } = data
    const taken = new Set([...state.picks, ...state.hidden, ...state.saved])

    let recommended = []
    let ranked = []
    if (state.picks.length) {
        const scores = scoreAll(unit, model.d, userVector(model, state.picks))
        // crude titles are never suggested, only found by searching for them
        ranked = topGames(scores, new Set([...taken, ...crude]), scores.length)
        recommended = (state.hideKnown ? ranked.filter((game) => rank[game] >= WELL_KNOWN) : ranked).slice(0, 10)
    }

    // under the search box: matches while typing, otherwise more games to pick
    let offered
    if (state.query.trim()) {
        offered = search(lowerNames, games.ratings, state.query, taken, 8)
        $('suggestions-label').textContent = offered.length ? 'Tap a game to add it:' : 'No game by that name. The data ends in 2021, so newer games are missing.'
    } else if (state.picks.length) {
        const shown = new Set(recommended)
        offered = ranked.filter((game) => rank[game] < FAMILIAR && !shown.has(game)).slice(0, 8)
        $('suggestions-label').textContent = 'Or tap one to add it. Similar to your picks, that you might have also played:'
    } else {
        offered = order.slice(0, 16)
        $('suggestions-label').textContent = 'Or tap a well-known game to add it:'
    }
    $('suggestions').replaceChildren(...offered.map((game) => chip(game, pick, 'add')))

    const n = state.picks.length
    $('picks-label').textContent = n ? `Your picks (${n})` : 'Your picks'
    $('picks').replaceChildren(...(n
        ? state.picks.map((game) => chip(game, hide, 'close'))
        : [el('span', { class: 'small-text' }, 'Nothing yet. Search for a game, tap a suggestion, or add a set.')]))
    $('clear').hidden = !n
    $('hidden-box').hidden = !state.hidden.length
    $('hidden-label').textContent = `Hidden from recommendations (${state.hidden.length})`
    $('hidden').replaceChildren(...state.hidden.map((game) => chip(game, restore, 'undo')))

    $('results').hidden = !n
    if (!n) return
    $('glance').textContent = glance(games, state.picks)
    $('status').textContent = n < 10
        ? `${n} pick${n === 1 ? '' : 's'} so far. The recommendations sharpen as you add more; ten is a good number.`
        : ''
    $('saved-label').textContent = state.saved.length
        ? 'Games you want to try. Tap one to take it off the list.'
        : 'Tap "Want to try" on a recommendation to keep it here.'
    $('saved').replaceChildren(...state.saved.map((game) => chip(game, unsave, 'bookmark')))
    $('recommendations').replaceChildren(...recommended.map(recommendation))

    const gem = hiddenGem(ranked, games, order, rank, state.picks, new Set([...taken, ...recommended]))
    $('gem').hidden = gem === undefined
    if (gem !== undefined) {
        $('gem').replaceChildren(
            el('nav', {},
                icon('diamond'),
                el('div', { class: 'max' },
                    el('div', { class: 'small-text' }, 'Hidden gem: your best match outside the 1,000 best-known games'),
                    el('a', { class: 'link', href: `https://boardgamegeek.com/boardgame/${games.id[gem]}`, target: '_blank', rel: 'noreferrer' },
                        el('h6', { class: 'small' }, title(gem))),
                    el('div', { class: 'small-text' }, details(gem)),
                    el('div', { class: 'small-text' },
                        `Rated ${games.average[gem].toFixed(1)} by ${games.ratings[gem].toLocaleString()} people. Closest to your pick ${games.name[closestPick(unit, model.d, gem, state.picks)]}`)),
                el('button', { class: 'border small', onclick: () => save(gem) }, icon('bookmark_add'), el('span', {}, 'Want to try')),
                el('button', { class: 'border small', onclick: () => hide(gem) }, icon('visibility_off'), el('span', {}, 'Not for me'))))
    }
}

// Light or dark is the portfolio's own setting: it is kept under the same name in
// the browser's storage, which the two share when they are on the same site.
function stored(mode) {
    try {
        if (mode) localStorage.setItem('mode', mode)
        return localStorage.getItem('mode')
    } catch {
        return null // storage can be blocked; the page then follows the system
    }
}
function theme(mode) {
    ui('mode', mode)
    const now = ui('mode') === 'dark' ? 'dark' : 'light'
    $('theme').firstElementChild.textContent = now === 'dark' ? 'light_mode' : 'dark_mode'
    // the portfolio's footer links and icons
    const links = {
        Handshake: 'https://msoe.joinhandshake.com/stu/users/55266622',
        LinkedIn: 'https://www.linkedin.com/in/billy-york/',
        GitHub: 'https://github.com/yorker78',
    }
    $('social').replaceChildren(...Object.entries(links).map(([name, href]) =>
        el('a', { href, target: '_blank', rel: 'noreferrer', ariaLabel: name, title: name },
            el('i', { class: 'tiny' }, el('img', { src: `https://www.billyyork.com/assets/img/${name.toLowerCase()}-${now}.png`, alt: '' })))))
    return now
}

async function start() {
    theme(stored() || 'auto')
    $('theme').addEventListener('click', () => theme(stored(ui('mode') === 'dark' ? 'light' : 'dark')))
    if (CODE_URL) Object.assign($('code-link'), { href: CODE_URL, hidden: false })
    try {
        data = await load()
    } catch (error) {
        $('loading').textContent = 'The model could not be loaded. Please try again later.'
        throw error
    }
    // a starter set fills in its games, most-rated edition of each name
    $('starters').replaceChildren(...Object.entries(STARTERS).map(([label, names]) => {
        const set = names.map((name) => data.order.find((game) => data.games.name[game] === name)).filter((game) => game !== undefined)
        // added to whatever is already picked
        return el('button', { class: 'chip fill', onclick: () => update(() => {
            state.picks = [...new Set([...state.picks, ...set])]
            state.hidden = state.hidden.filter((game) => !set.includes(game))
            state.saved = state.saved.filter((game) => !set.includes(game))
        }) }, icon('add'), el('span', {}, label))
    }))
    $('search').addEventListener('input', (event) => update(() => (state.query = event.target.value)))
    $('hide-known').addEventListener('change', (event) => update(() => (state.hideKnown = event.target.checked)))
    $('clear').addEventListener('click', () => update(() => (state.picks = [])))
    $('copy').addEventListener('click', async () => {
        await navigator.clipboard.writeText(location.href)
        $('copy').lastElementChild.textContent = 'Link copied'
    })
    recall()
    $('loading').hidden = true
    draw()
}

// BeerCSS's script sets `ui` when it has loaded; modules run in order, so it is ready here
start()
