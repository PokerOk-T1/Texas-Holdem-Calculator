#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import json
import math
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Optional, Literal

# -----------------------------
# Карты и парсинг
# -----------------------------

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

RANK_CHAR_TO_VALUE = {ch: i + 2 for i, ch in enumerate(RANK_CHARS)}
VALUE_TO_RANK_CHAR = {v: k for k, v in RANK_CHAR_TO_VALUE.items()}
RANK_TO_INDEX = {ch: i for i, ch in enumerate(RANK_CHARS)}


@dataclass(frozen=True)
class Card:
    rank: int  # 2..14
    suit: int  # 0..3 (c, d, h, s)

    def __str__(self) -> str:
        return VALUE_TO_RANK_CHAR[self.rank] + SUIT_CHARS[self.suit]


def parse_card(s: str) -> Card:
    """Парсинг одной карты вида 'As', 'Td' и т.п."""
    s = s.strip()
    if len(s) != 2:
        raise ValueError(f"Invalid card format: {s!r}")
    r, su = s[0].upper(), s[1].lower()
    if r not in RANK_CHAR_TO_VALUE or su not in SUIT_CHARS:
        raise ValueError(f"Invalid card: {s!r}")
    return Card(RANK_CHAR_TO_VALUE[r], SUIT_CHARS.index(su))


def parse_cards(s: str) -> List[Card]:
    """
    Парсинг строки карт: 'As Kd' или 'AsKd' -> список Card.
    """
    s = s.strip()
    if not s:
        return []
    parts = s.split()
    if len(parts) == 1 and len(s) % 2 == 0 and len(s) > 2:
        parts = [s[i:i + 2] for i in range(0, len(s), 2)]
    return [parse_card(p) for p in parts]


# -----------------------------
# Оценка покерных рук
# -----------------------------

def evaluate_5cards(cards: List[Card]) -> int:
    """
    Оценка силы 5-карточной руки.
    Возвращает int, где большее значение = более сильная рука.
    Кодировка: 4 бита на категорию и на каждый ранг (всего 8 "нибблов").
    Категории:
      8 - стрит-флеш
      7 - каре
      6 - фулл-хаус
      5 - флеш
      4 - стрит
      3 - сет
      2 - две пары
      1 - пара
      0 - старшая карта
    """
    assert len(cards) == 5
    ranks = sorted((c.rank for c in cards), reverse=True)
    suits = [c.suit for c in cards]

    counts = collections.Counter(ranks)
    items = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))

    flush = len(set(suits)) == 1
    unique_ranks = sorted(set(ranks), reverse=True)

    # Стрит
    straight = False
    straight_high = 0
    if len(unique_ranks) >= 5:
        for i in range(len(unique_ranks) - 4 + 1):
            window = unique_ranks[i:i + 5]
            if len(window) == 5 and window[0] - window[4] == 4:
                straight = True
                straight_high = window[0]
                break
    # Вилочное колесо A-5
    if not straight:
        wheel = [14, 5, 4, 3, 2]
        if all(r in ranks for r in wheel):
            straight = True
            straight_high = 5

    # Категория
    if straight && flush:
        cat = 8
        primary = [straight_high]
    elif items[0][1] == 4:
        quad = items[0][0]
        kicker = max(r for r in ranks if r != quad)
        cat = 7
        primary = [quad, kicker]
    elif items[0][1] == 3 and items[1][1] >= 2:
        trip = items[0][0]
        pair = items[1][0]
        cat = 6
        primary = [trip, pair]
    elif flush:
        cat = 5
        primary = sorted(ranks, reverse=True)
    elif straight:
        cat = 4
        primary = [straight_high]
    elif items[0][1] == 3:
        trip = items[0][0]
        kickers = [r for r in ranks if r != trip]
        kickers = sorted(kickers, reverse=True)[:2]
        cat = 3
        primary = [trip] + kickers
    elif items[0][1] == 2 and items[1][1] == 2:
        pair1 = max(items[0][0], items[1][0])
        pair2 = min(items[0][0], items[1][0])
        kicker = max(r for r in ranks if r != pair1 and r != pair2)
        cat = 2
        primary = [pair1, pair2, kicker]
    elif items[0][1] == 2:
        pair = items[0][0]
        kickers = [r for r in ranks if r != pair]
        kickers = sorted(kickers, reverse=True)[:3]
        cat = 1
        primary = [pair] + kickers
    else:
        cat = 0
        primary = sorted(unique_ranks, reverse=True)[:5]

    values = [cat] + primary
    values += [0] * (8 - len(values))  # доводим до 1 категории + 7 рангов

    v = 0
    for x in values:
        v = (v << 4) | x
    return v


def evaluate_7cards(cards: List[Card]) -> int:
    """
    Оценка силы 7-карточной руки:
    перебор всех 5-карточных комбинаций и выбор максимальной.
    """
    best = -1
    for combo in combinations(cards, 5):
        val = evaluate_5cards(list(combo))
        if val > best:
            best = val
    return best


# -----------------------------
# Колода и стартовые руки
# -----------------------------

FULL_DECK: List[Card] = [Card(rank, suit) for rank in range(2, 15) for suit in range(4)]

ALL_STARTING_HANDS: List[Tuple[Card, Card]] = []
for i in range(len(FULL_DECK)):
    for j in range(i + 1, len(FULL_DECK)):
        c1, c2 = FULL_DECK[i], FULL_DECK[j]
        ALL_STARTING_HANDS.append((c1, c2))


def combo_descriptor(c1: Card, c2: Card) -> Tuple[str, str, str]:
    """
    Описание стартовой руки:
      (старший ранг, младший ранг, suited_flag)
    где suited_flag: '' для пары, 's' для одномастной, 'o' для разномастной.
    """
    if c1.rank == c2.rank:
        r1 = r2 = VALUE_TO_RANK_CHAR[c1.rank]
    else:
        if c1.rank > c2.rank:
            hi, lo = c1, c2
        else:
            hi, lo = c2, c1
        r1 = VALUE_TO_RANK_CHAR[hi.rank]
        r2 = VALUE_TO_RANK_CHAR[lo.rank]
    if c1.rank == c2.rank:
        suited = ''
    else:
        suited = 's' if c1.suit == c2.suit else 'o'
    return r1, r2, suited


# -----------------------------
# Диапазоны рук
# -----------------------------

@dataclass
class RangeToken:
    first: str
    second: str
    suited: Optional[str]  # 's', 'o' или None (любой)
    plus: bool

    @property
    def is_pair(self) -> bool:
        return self.first == self.second


def parse_range_token(s: str) -> RangeToken:
    """
    Парсинг токена диапазона, примеры:
      '22', '55+', 'A2s', 'A2s+', 'KTo', 'KQo+', 'AK', 'AQ+'
    """
    t = s.strip().replace(" ", "").upper()
    if not t:
        raise ValueError("Empty range token")
    plus = t.endswith('+')
    if plus:
        t = t[:-1]
    suited = None
    if t and t[-1] in ('S', 'O'):
        suited = t[-1].lower()
        t = t[:-1]
    if len(t) != 2 or t[0] not in RANK_CHARS or t[1] not in RANK_CHARS:
        raise ValueError(f"Invalid range token: {s}")
    first, second = t[0], t[1]
    return RangeToken(first, second, suited, plus)


def combo_matches_token(c1: Card, c2: Card, token: RangeToken) -> bool:
    r1, r2, suited_flag = combo_descriptor(c1, c2)

    if token.is_pair:
        # Пары
        if r1 != token.first or r1 != r2:
            return False
        if token.plus:
            # Любая пара с рангом >= заданного
            return RANK_TO_INDEX[r1] >= RANK_TO_INDEX[token.first]
        else:
            return True

    # Непарные руки
    if r1 == r2:
        return False

    if token.suited is not None and suited_flag != token.suited:
        return False

    if not token.plus:
        return r1 == token.first and r2 == token.second
    else:
        # first фиксирован, second от заданного до first-1
        if r1 != token.first:
            return False
        v2 = RANK_TO_INDEX[r2]
        v_second = RANK_TO_INDEX[token.second]
        v_first = RANK_TO_INDEX[token.first]
        return v_second <= v2 < v_first


def expand_range_to_combos(range_str: str) -> Tuple[List[RangeToken], List[Tuple[Card, Card]]]:
    """
    Преобразует строку диапазона в список токенов и список конкретных стартовых комбинаций.
    """
    tokens_raw = [t for t in range_str.split(',') if t.strip()]
    tokens: List[RangeToken] = []
    for tr in tokens_raw:
        try:
            tokens.append(parse_range_token(tr))
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)

    combos: List[Tuple[Card, Card]] = []
    for c1, c2 in ALL_STARTING_HANDS:
        for tok in tokens:
            if combo_matches_token(c1, c2, tok):
                combos.append((c1, c2))
                break
    return tokens, combos


# -----------------------------
# Оппонент и колода
# -----------------------------

@dataclass
class Opponent:
    kind: Literal['hand', 'range']
    cards: Optional[List[Card]] = None
    range_str: Optional[str] = None
    range_tokens: Optional[List[RangeToken]] = None
    range_combos: Optional[List[Tuple[Card, Card]]] = None


def build_deck(excluded: List[Card]) -> List[Card]:
    excluded_set = set(excluded)
    return [c for c in FULL_DECK if c not in excluded_set]


# -----------------------------
# Симуляция (exact / Monte Carlo)
# -----------------------------

def can_use_exact(board: List[Card], opponents: List[Opponent]) -> bool:
    """
    Проверка, можно ли использовать полный перебор:
    - все оппоненты с фиксированными руками,
    - не больше 2 оппонентов (3 игрока всего),
    - на борде не больше 3 неизвестных карт,
    - общее число комбинаций не превышает 1e6.
    """
    if any(o.kind != 'hand' for o in opponents):
        return False
    if len(opponents) > 2:
        return False
    known_board = len(board)
    missing = 5 - known_board
    if missing < 0 or missing > 3:
        return False
    remaining_cards = 52 - (2 + known_board + 2 * len(opponents))
    if remaining_cards < missing:
        return False
    combos = math.comb(remaining_cards, missing)
    return combos <= 1_000_000


def simulate_exact(hero: List[Card], board: List[Card], opponents: List[Opponent]) -> dict:
    """Точный перебор всех возможных недостающих карт борда."""
    dead = hero + board
    for o in opponents:
        dead.extend(o.cards)
    deck = build_deck(dead)
    missing = 5 - len(board)
    if missing == 0:
        boards = [board]
    else:
        boards = [board + list(extra) for extra in combinations(deck, missing)]

    wins = ties = losses = 0
    for full_board in boards:
        hero_score = evaluate_7cards(hero + full_board)
        opp_scores = [evaluate_7cards(o.cards + full_board) for o in opponents]
        all_scores = [hero_score] + opp_scores
        max_score = max(all_scores)
        winners = [i for i, s in enumerate(all_scores) if s == max_score]
        if 0 in winners:
            if len(winners) == 1:
                wins += 1
            else:
                ties += 1
        else:
            losses += 1

    total = wins + ties + losses
    return {"wins": wins, "ties": ties, "losses": losses, "iterations": total}


def simulate_mc(
    hero: List[Card],
    board: List[Card],
    opponents: List[Opponent],
    iterations: int,
    rng: random.Random
) -> dict:
    """Симуляция Монте-Карло."""
    wins = ties = losses = 0
    effective = 0

    for _ in range(iterations):
        fixed_opps = [o for o in opponents if o.kind == 'hand']
        range_opps = [o for o in opponents if o.kind == 'range']

        dead = list(hero) + list(board)
        for o in fixed_opps:
            dead.extend(o.cards)

        opp_hands: List[List[Card]] = []
        failed = False

        # Выбор рук из диапазонов
        for o in range_opps:
            dead_set = set(dead)
            available = []
            for c1, c2 in o.range_combos:
                if c1 not in dead_set and c2 not in dead_set:
                    available.append((c1, c2))
            if not available:
                failed = True
                break
            hand = available[rng.randrange(len(available))]
            opp_hands.append(list(hand))
            dead.extend(hand)

        if failed:
            continue

        # Добавляем фиксированные руки
        for o in fixed_opps:
            opp_hands.append(o.cards)

        # Добираем борд
        missing = 5 - len(board)
        full_board = list(board)
        if missing > 0:
            deck = build_deck(dead)
            if len(deck) < missing:
                continue
            extra = rng.sample(deck, missing)
            full_board.extend(extra)

        hero_score = evaluate_7cards(hero + full_board)
        scores = [hero_score] + [evaluate_7cards(h + full_board) for h in opp_hands]
        max_score = max(scores)
        winners = [i for i, s in enumerate(scores) if s == max_score]

        if 0 in winners:
            if len(winners) == 1:
                wins += 1
            else:
                ties += 1
        else:
            losses += 1

        effective += 1

    return {"wins": wins, "ties": ties, "losses": losses, "iterations": effective}


def simulate(
    hero: List[Card],
    board: List[Card],
    opponents: List[Opponent],
    iterations: int,
    mode: str,
    seed: Optional[int] = None
) -> dict:
    """Обёртка над exact/MC, возвращает готовую статистику."""
    rng = random.Random(seed)
    use_exact = (mode == 'exact' and can_use_exact(board, opponents))

    if mode == 'exact' and not use_exact:
        # просто fallback, сообщение отдаём на уровень выше
        pass

    if use_exact:
        raw = simulate_exact(hero, board, opponents)
        mode_used = 'exact'
    else:
        raw = simulate_mc(hero, board, opponents, iterations, rng)
        mode_used = 'mc'

    iters = max(1, raw["iterations"])
    win = raw["wins"] / iters
    tie = raw["ties"] / iters
    lose = raw["losses"] / iters
    approx_error = 1 / math.sqrt(iters)

    return {
        "mode_used": mode_used,
        "iterations": iters,
        "win": win,
        "tie": tie,
        "lose": lose,
        "approx_error": approx_error,
    }


# -----------------------------
# Self-test
# -----------------------------

def self_test() -> bool:
    """Простейшие встроенные тесты корректности."""
    ok = True

    def check(name: str, cond: bool):
        nonlocal ok
        if cond:
            print(f"[OK]   {name}")
        else:
            ok = False
            print(f"[FAIL] {name}")

    # Парсинг карт
    c = parse_cards("As Kd")
    check("parse_cards", len(c) == 2 and str(c[0]) == "As" and str(c[1]) == "Kd")

    # AA vs KK префлоп: эквити AA > 0.75
    hero = parse_cards("As Ad")
    vill = Opponent(kind='hand', cards=parse_cards("Kc Kd"))
    res = simulate(hero, [], [vill], iterations=5000, mode='mc', seed=123)
    check("AA vs KK equity", res["win"] > 0.75)

    # Одинаковые руки -> почти всегда ничья
    hero = parse_cards("As Ad")
    vill = Opponent(kind='hand', cards=parse_cards("As Ad"))
    res = simulate(hero, [], [vill], iterations=3000, mode='mc', seed=123)
    check("identical hands tie", res["tie"] > 0.9)

    # Нутсовый флеш против более низкого флеша на уже готовом борде -> почти 100% победа
    board = parse_cards("Ah Kh Qh Jh 2c")
    hero = parse_cards("Th 3c")   # стрит-флеш до T
    vill = Opponent(kind='hand', cards=parse_cards("9h 9c"))
    res = simulate(hero, board, [vill], iterations=1, mode='exact')
    check("nuts flush vs lower flush", res["win"] > 0.99)

    if ok:
        print("Self-test passed")
    else:
        print("Self-test FAILED")
    return ok


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Texas Hold'em equity calculator")
    p.add_argument("--hero", "-H", required=False, help="Hero hand, e.g. 'As Kd'")
    p.add_argument("--board", "-B", default="", help="Board cards, e.g. '2c 7d Jh'")
    p.add_argument(
        "--villain-hand",
        action="append",
        dest="villain_hands",
        help="Villain hand, can be used multiple times"
    )
    p.add_argument(
        "--villain-range",
        action="append",
        dest="villain_ranges",
        help="Villain range, e.g. '22+,A2s+,KTs+,QJo+'"
    )
    p.add_argument(
        "--iterations", "-N",
        type=int,
        default=100000,
        help="Monte Carlo iterations (default: 100000)"
    )
    p.add_argument(
        "--mode",
        choices=["mc", "exact"],
        default="mc",
        help="Calculation mode: mc (default) or exact"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Monte Carlo"
    )
    p.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help="Output JSON instead of human-readable text"
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal tests and exit"
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        ok = self_test()
        sys.exit(0 if ok else 1)

    if not args.hero:
        parser.error("--hero is required unless --self-test is used")

    hero_cards = parse_cards(args.hero)
    if len(hero_cards) != 2:
        parser.error("Hero must have exactly 2 cards")

    board_cards = parse_cards(args.board) if args.board else []
    if len(board_cards) > 5:
        parser.error("Board cannot have more than 5 cards")

    opponents: List[Opponent] = []
    seen_cards = set(hero_cards + board_cards)

    # Фиксированные руки оппонентов
    if args.villain_hands:
        for vs in args.villain_hands:
            cards = parse_cards(vs)
            if len(cards) != 2:
                parser.error(f"Villain hand '{vs}' must contain exactly 2 cards")
            for c in cards:
                if c in seen_cards:
                    parser.error(f"Duplicate card detected in villain hand: {c}")
                seen_cards.add(c)
            opponents.append(Opponent(kind='hand', cards=cards))

    # Диапазоны оппонентов
    if args.villain_ranges:
        for rs in args.villain_ranges:
            tokens, combos = expand_range_to_combos(rs)
            if not combos:
                parser.error(f"Villain range '{rs}' produced no valid combos")
            opponents.append(
                Opponent(
                    kind='range',
                    range_str=rs,
                    range_tokens=tokens,
                    range_combos=combos
                )
            )

    if not opponents:
        parser.error("At least one --villain-hand or --villain-range must be specified")

    # Запуск расчёта
    result = simulate(
        hero_cards,
        board_cards,
        opponents,
        iterations=args.iterations,
        mode=args.mode,
        seed=args.seed
    )

    # Подготовка вывода
    opp_info = []
    for o in opponents:
        if o.kind == 'hand':
            opp_info.append({"type": "hand", "cards": [str(c) for c in o.cards]})
        else:
            opp_info.append(
                {
                    "type": "range",
                    "raw": o.range_str,
                    "combo_count": len(o.range_combos),
                }
            )

    output = {
        "hero": [str(c) for c in hero_cards],
        "board": [str(c) for c in board_cards],
        "opponents": opp_info,
        "mode": result["mode_used"],
        "iterations": result["iterations"],
        "results": {
            "win": result["win"],
            "tie": result["tie"],
            "lose": result["lose"],
        },
        "approx_error": result["approx_error"],
    }

    if args.json_out:
        print(json.dumps(output, indent=2))
    else:
        print(f"Hero: {' '.join(output['hero'])}")
        print(f"Board: {' '.join(output['board']) if output['board'] else '(none)'}")

        print("\nOpponents:")
        for idx, o in enumerate(opp_info, start=1):
            if o["type"] == "hand":
                print(f"  #{idx} hand: {' '.join(o['cards'])}")
            else:
                print(f"  #{idx} range: {o['raw']}   ({o['combo_count']} combos)")

        if args.mode == "exact" and result["mode_used"] != "exact":
            print("\nNote: exact mode was requested but not feasible; Monte Carlo was used instead.")

        print(f"\nMode: {'Exact' if result['mode_used'] == 'exact' else 'Monte Carlo'}")
        print(f"Iterations: {result['iterations']}")

        print("\nResults:")
        print(f"  Hero win:   {result['win'] * 100:6.2f} %")
        print(f"  Tie:        {result['tie'] * 100:6.2f} %")
        print(f"  Hero lose:  {result['lose'] * 100:6.2f} %")

        print(f"\nApprox. error (sigma): ~{result['approx_error'] * 100:4.2f} %")


if __name__ == "__main__":
    main()