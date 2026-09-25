from app.text import build_segments, clean, speaker_at, to_srt


def test_clean_removes_fillers_and_fixes_case():
    assert clean("Ну, по бэкенду э-э всё готово, как бы.") == "По бэкенду всё готово."
    assert clean("вот такие дела") == "Такие дела"
    # words that merely contain a filler stay intact
    assert clean("Нужно вотировать и типажи") == "Нужно вотировать и типажи"
    # never return an empty string
    assert clean("Ну.") == "Ну."


def test_speaker_at_picks_max_overlap_and_nearest():
    turns = [{"speaker": "A", "start": 0, "end": 5}, {"speaker": "B", "start": 4, "end": 10}]
    assert speaker_at(turns, 3, 4.5) == "A"
    assert speaker_at(turns, 4.2, 6) == "B"
    assert speaker_at(turns, 11, 12) == "B"
    assert speaker_at([], 1, 2) is None


def w(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_build_segments_splits_on_speaker_and_pause():
    turns = [{"speaker": "A", "start": 0, "end": 3}, {"speaker": "B", "start": 3, "end": 20}]
    words = [w(" Привет,", 0.1, 0.5), w(" ну", 0.6, 0.8), w(" всем.", 0.9, 1.4),
             w(" Здравствуйте.", 3.2, 4.0),
             w(" Продолжим.", 7.0, 7.8)]  # long pause -> new segment, same speaker
    segs = build_segments(words, turns)
    assert [s["speaker"] for s in segs] == ["A", "B", "B"]
    assert segs[0]["text"] == "Привет, ну всем."
    assert segs[0]["text_clean"] == "Привет, всем."
    assert (segs[0]["start"], segs[0]["end"]) == (0.1, 1.4)


def test_srt_format():
    srt = to_srt([{"speaker": "Анна", "start": 3661.5, "end": 3662.25, "text": "Да."}])
    assert srt == "1\n01:01:01,500 --> 01:01:02,250\nАнна: Да.\n"


def test_phrases_split_on_sentence_end_gap_and_signal_pause():
    from app.ml import phrases

    words = [w(" Так,", 0.0, 0.3), w(" привет.", 0.35, 0.8), w(" Да", 0.9, 1.1), w(" конечно", 1.15, 1.6),
             w(" потом", 2.2, 2.5), w(" ещё", 2.55, 2.8)]
    assert phrases(words) == [(0.0, 0.8), (0.9, 1.6), (2.2, 2.8)]
    # a pause detected in the signal between two words splits even when timings touch
    assert phrases(words, pauses=[2.67]) == [(0.0, 0.8), (0.9, 1.6), (2.2, 2.5), (2.55, 2.8)]


def test_silences_and_merge_turns():
    import numpy as np

    from app.ml import merge_turns, silences

    voiced = np.array([True] * 50 + [False] * 20 + [True] * 50 + [False] * 5 + [True] * 10)
    assert silences(voiced) == [1.2]  # only the 0.4 s pause, not the 0.1 s one
    turns = merge_turns([(0, 1), (1.2, 2), (2.5, 3)], [0, 0, 1])
    assert turns == [{"speaker": "SPEAKER_00", "start": 0, "end": 2}, {"speaker": "SPEAKER_01", "start": 2.5, "end": 3}]


def test_label_phrases_short_phrase_follows_neighbour():
    import numpy as np

    from app.ml import label_phrases

    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    mixed = np.array([0.7, 0.71, 0])  # ambiguous short phrase
    mixed /= np.linalg.norm(mixed)
    emb = np.stack([a, a, b, mixed, b, a])
    spans = [(0, 2), (2.1, 4), (4.5, 6), (6.05, 6.5), (6.6, 8), (9, 11)]
    voiced = np.array([2, 1.9, 1.5, 0.45, 1.4, 2])
    assert label_phrases(emb, spans, voiced, num_speakers=2) == [0, 0, 1, 1, 1, 0]


def test_verify_names_needs_evidence_from_transcript():
    from app.text import verify_names

    turns = [("A", "Всем привет."), ("A", "Игорь, сколько тебе нужно?"), ("B", "Неделя."),
             ("C", "Меня зовут Олег, я из маркетинга."), ("A", "Марина, сделай макет."), ("C", "Бюджет такой.")]
    props = [{"label": "B", "name": "Игорь"}, {"label": "D", "name": "Игорь"},
             {"label": "C", "name": "Олег"}, {"label": "C", "name": "Марина"}, {"label": "A", "name": "Анна"}]
    assert verify_names(props, turns) == {"B": "Игорь", "C": "Олег"}


def test_verify_names_ignores_name_pushed_to_turn_end():
    from app.text import verify_names

    # ASR timing moved «Игорь,» to the end of Marina's turn; Anna speaks next.
    turns = [("M", "Я нарисую экран. Игорь,"), ("A", "Сколько тебе нужно времени?"), ("I", "Неделя.")]
    assert verify_names([{"label": "A", "name": "Игорь"}], turns) == {}


def test_verify_names_counts_questions_not_instructions():
    from app.text import verify_names

    # After an instruction the next speaker is often someone else; only a question expects an answer.
    turns = [("A", "Марина, сделай три экрана до пятницы."), ("O", "Бюджет на октябрь — триста тысяч."),
             ("A", "Дмитрий, как думаешь, можно разбить работу?"), ("D", "Да, на две очереди.")]
    props = [{"label": lab, "name": n} for n in ("Марина", "Дмитрий") for lab in "AOD"]
    assert verify_names(props, turns) == {"D": "Дмитрий"}
