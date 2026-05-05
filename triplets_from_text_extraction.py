import json
import re
from typing import List
import stanza
stanza_pipeline = stanza.Pipeline("ru", processors="tokenize,pos,lemma,depparse")

REL_PRON_FORMS = {"который", "которая", "которое", "которые"}
SKIP_DEPRELS = {"punct", "cc", "case", "cop", "aux", "mark"}
SKIP_UPOS = {"PUNCT", "CCONJ", "SCONJ", "ADP", "AUX", "PART"}


def replace_pron(subj, sent):
    sent_prev_nouns = [word for word in sent.words if word.upos == "NOUN" and word.id < subj.id][::-1]
    if subj.text.endswith(("ый", "ым", "ому")):
        for i in range(len(sent_prev_nouns)):
            if has_feat(sent_prev_nouns[i], "Gender", "Masc") and has_feat(sent_prev_nouns[i], "Number", "Sing"):
                return sent_prev_nouns[i]
    if subj.text.endswith(("ая", "ой")):
        for i in range(len(sent_prev_nouns)):
            if has_feat(sent_prev_nouns[i], "Gender", "Fem") and has_feat(sent_prev_nouns[i], "Number", "Sing"):
                return sent_prev_nouns[i]
    if subj.text.endswith(("ое")):
        for i in range(len(sent_prev_nouns)):
            if has_feat(sent_prev_nouns[i], "Gender", "Neut") and has_feat(sent_prev_nouns[i], "Number", "Sing"):
                return sent_prev_nouns[i]
    if subj.text.endswith(("ые", "ым")):
        for i in range(len(sent_prev_nouns)):
            if has_feat(sent_prev_nouns[i], "Number", "Plur"):
                return sent_prev_nouns[i]
    return subj



def pretty_print(word_list):
    for word in word_list:
        print(word.text, end="; ")
    print()

def find_verbs(sent):
    verbs = []
    for word in sent.words:
        if word.upos == "VERB":
            verbs.append(word)
    return verbs

def find_nouns(sent):
    nouns = []
    for word in sent.words:
        if word.upos in ("NOUN", "PRON", "PROPN"):
            nouns.append(word)
    return nouns

def is_predicate_word(word):
    if word.upos == "VERB":
        if has_feat(word, "VerbForm", "Conv"):
            return False
        if has_feat(word, "VerbForm", "Part") and word.deprel in ("amod", "acl"):
            return False
        return True
    if word.upos == "ADJ" and has_feat(word, "Variant", "Short"):
        return True

    return False

def find_predicates(sent):
    return [word for word in sent.words if is_predicate_word(word)]

def has_feat(word, feat_name, feat_value=None):
    if not word.feats:
        return False
    for item in word.feats.split("|"):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        if k == feat_name and (feat_value is None or v == feat_value):
            return True
    return False


def get_children(sent, head_id):
    return [w for w in sent.words if w.head == head_id]


def has_case_marker(sent, word):
    for child in get_children(sent, word.id):
        if child.deprel == "case":
            return True
    return False

def get_preposition(sent, word):
    for child in get_children(sent, word.id):
        if child.deprel == "case":
            return child
    return None


def is_subject_like(noun, predicate):
    return (noun.head == predicate.id and (not has_feat(noun, "Case", "Acc")) and noun.deprel in (
        "subj", "nsubj", "nsubj:pass", "csubj"
    ))

def find_root_subject(sent):
    for word in sent.words:
        if (word.deprel == "root" and word.upos in ("NOUN", "PROPN", "PRON")):
            return word
    return None

def build_root_subj_fallback_triplets(sent, include_full_words=False):
    root_subj = find_root_subject(sent)
    if root_subj is None:
        return []
    root_noun_children = [word for word in get_children(sent, root_subj.id) if word.upos in ("NOUN", "PRON", "PROPN")]
    result = [{
        "subject": root_subj if include_full_words else root_subj.text,
        "predicate": "",
        "object": ""
    }]
    for word in root_noun_children:
        if word.deprel == "conj":
            result.append({
                "subject": word if include_full_words else word.text,
                "predicate": "",
                "object": ""
            })
    # print(result)
    return result


def is_object_like(sent, noun, predicate):
    if noun.head == predicate.id and noun.deprel in ("obj", "iobj"):
        return True

    if (noun.head == predicate.id and noun.upos == "PROPN") or (
        noun.head == predicate.id
        and (noun.upos in ("NOUN", "PRON"))
        and noun.deprel == "obl"
        and (not has_case_marker(sent, noun) or get_preposition(sent, noun).text.lower() in ('на', ))):
        return True
    if noun.head == predicate.id and noun.deprel == "nsubj" and has_feat(noun, "Case", "Acc"):
        return True
    return False


def subject_fallback(nouns, subj):
    result = []
    stack = [subj]
    visited = set()

    while stack:
        current = stack.pop()
        if current.id in visited:
            continue
        visited.add(current.id)

        for noun in nouns:
            if noun.head == current.id and noun.deprel in ("conj", "parataxis"): #
                result.append(noun)
                stack.append(noun)
            elif noun.head == current.id and noun.deprel == "flat:foreign" and noun.upos == "PROPN":
                result.append(noun)
                stack.append(noun)
    return result

def object_fallback(nouns, obj):
    result = []
    stack = [obj]
    visited = set()

    while stack:
        current = stack.pop()
        if current.id in visited:
            continue
        visited.add(current.id)

        for noun in nouns:
            if noun.head == current.id and noun.deprel in ("conj", "parataxis"):
                result.append(noun)
                stack.append(noun)

    return result


def is_short_adj(word):
    return word.upos == "ADJ" and has_feat(word, "Variant", "Short")


def norm_dep(deprel):
    if deprel in ("obj", "iobj", "conj"):
        return deprel
    return None


def compatible_predicates(head_pred, conj_pred):
    
    if ((is_short_adj(head_pred) and not is_short_adj(conj_pred))
        or (is_short_adj(conj_pred) and not is_short_adj(head_pred))):
        return False

    return True


def can_share_candidate(head_pred, conj_pred, candidate, words, subj_and_obj_for_predicate):
    conj_compls = subj_and_obj_for_predicate[conj_pred]
    if conj_compls:
        return False

    if not compatible_predicates(head_pred, conj_pred):
        return False

    if not norm_dep(candidate.deprel):
        return False

    return True


def find_triplets(sent, nouns, predicates):
    ids_predicates = {pred.id:pred for pred in predicates}
    subj_and_obj_for_predicate = {}
    for predicate in predicates:
        predicate_subjects = []
        predicate_objects = []
        for noun in nouns:
            if is_subject_like(noun, predicate):
                predicate_subjects.append(noun)
            elif is_object_like(sent, noun, predicate):
                predicate_objects.append(noun)

        subj_and_obj_for_predicate[predicate] = {"subjects": predicate_subjects,
                                                 "objects": predicate_objects}
    for predicate in predicates:
        if predicate.head in ids_predicates:
            head_predicate = ids_predicates[predicate.head]

            if not subj_and_obj_for_predicate[predicate]["subjects"]:
                predicate_subjects = subj_and_obj_for_predicate[head_predicate]["subjects"]
                subj_and_obj_for_predicate[predicate]["subjects"] += predicate_subjects

            if predicate.deprel == "conj" and not subj_and_obj_for_predicate[predicate]["objects"]:
                head_objects = subj_and_obj_for_predicate[head_predicate]["objects"]
                shared = [
                    obj for obj in head_objects
                    if can_share_candidate(head_predicate, predicate, obj, sent.words, subj_and_obj_for_predicate)
                ]
                subj_and_obj_for_predicate[predicate]["objects"] += shared
    
    for predicate in predicates:
        subjects = subj_and_obj_for_predicate[predicate]["subjects"]
        objects = subj_and_obj_for_predicate[predicate]["objects"]
        
        for subj in subjects:
            addit_subj = subject_fallback(nouns, subj)
            subj_and_obj_for_predicate[predicate]["subjects"] += addit_subj
        
        for obj in objects:
            addit_obj = object_fallback(nouns, obj)
            for x in addit_obj:
                if x not in subj_and_obj_for_predicate[predicate]["objects"]:
                    subj_and_obj_for_predicate[predicate]["objects"].append(x)
    for predicate in predicates:
        subjects = subj_and_obj_for_predicate[predicate]["subjects"]
        objects = subj_and_obj_for_predicate[predicate]["objects"]
        for i, s in enumerate(subjects):
            if s.text in REL_PRON_FORMS:
                # print(s)
                new_s = replace_pron(s, sent)
                # print(new_s)
                subj_and_obj_for_predicate[predicate]["subjects"][i] = new_s
        for i, o in enumerate(objects):
            if o.text in REL_PRON_FORMS:
                new_o = replace_pron(o, sent)
                subj_and_obj_for_predicate[predicate]["objects"][i] = new_o

    
    return subj_and_obj_for_predicate

def deduplicate_dicts(items):
    seen = set()
    result = []

    for item in items:
        key = tuple(sorted(item.items()))
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result

def format_triplets(subj_and_obj_for_predicate, include_full_words=False, mode='separate'):
    triplets = []
    for pred, subj_obj in subj_and_obj_for_predicate.items():
        predicate = pred.text if not include_full_words else pred
        if pred.text.lower() in ('мочь', 'могут', 'может', 'можно', 'должна', 'должны', 'должен'):
            continue
        elif subj_obj["subjects"] and subj_obj["objects"]:
            for s in subj_obj["subjects"]:
                for o in subj_obj["objects"]:
                    tripl_dict = {"subject": s.text if not include_full_words else s,
                                    "predicate": predicate,
                                    "object": o.text if not include_full_words else o}
                    triplets.append(tripl_dict)
        elif subj_obj["subjects"] and not subj_obj["objects"]:
            for s in subj_obj["subjects"]:
                tripl_dict = {"subject": s.text if not include_full_words else s,
                                  "predicate": predicate,
                                  "object": ""}
                triplets.append(tripl_dict)
        elif not subj_obj["subjects"] and subj_obj["objects"]:
            for o in subj_obj["objects"]:
                tripl_dict = {"subject": "",
                                "predicate": predicate,
                                "object": o.text if not include_full_words else o}
                triplets.append(tripl_dict)
        
        elif mode == 'separate' and not subj_obj["subjects"] and not subj_obj["objects"]:
            tripl_dict = {"subject": "",
                            "predicate": predicate,
                            "object": ""}
            triplets.append(tripl_dict)
        
    return deduplicate_dicts(triplets)


def build_json_extracted(sent, triplets):
    return {"sentence": sent.text,
            "triplets": triplets}

def extract(text, include_full_words=False):
    result = []
    document = stanza_pipeline(text)
    for sent in document.sentences:
        nouns = find_nouns(sent)
        verbs = find_verbs(sent)
        predicates = find_predicates(sent)
        subj_and_obj_for_predicate = find_triplets(sent, nouns, predicates)
        triplets = format_triplets(subj_and_obj_for_predicate, include_full_words)
        result.append(build_json_extracted(sent, triplets))
    return result

def extract_one_sentence(sentence, include_full_words=False, mode='separate'):
    sentence_init = sentence
    sentence = re.sub(r" таки(?:е|ми|м|й|ая|ое) как", ":", sentence)

    doc = stanza_pipeline(sentence)
    if not doc.sentences:
        return {"sentence": sentence_init, "sent_obj": None, "triplets": []}
    
    sent = doc.sentences[0]
    root_triplets = None
    if mode == 'separate':
        root_triplets = build_root_subj_fallback_triplets(sent, include_full_words)
    if root_triplets:
        return {
            "sentence": sentence_init,
            "sent_obj": sent,
            "triplets": root_triplets
        }
    #print(sent)
    nouns = find_nouns(sent)
    predicates = find_predicates(sent)
    subj_and_obj_for_predicate = find_triplets(sent, nouns, predicates)
    triplets = format_triplets(subj_and_obj_for_predicate, include_full_words, mode=mode)
    # if not triplets:
    #     triplets = build_root_subj_fallback_triplets(sent, include_full_words)
    return {
        "sentence": sentence_init,
        "sent_obj": sent,
        "triplets": triplets
    }


def is_stanza_word(x):
    return hasattr(x, "id") and hasattr(x, "text")


def should_keep_child(child, role):
    if child.deprel in SKIP_DEPRELS:
        return False
    if child.upos in SKIP_UPOS:
        return False

    return True


def frame_node_text(word):
    if not is_stanza_word(word):
        return str(word)
    if word.upos == "ADJ" or (word.upos == "VERB" and has_feat(word, "VerbForm", "Part")):
        return word.text.lower()
    return word.lemma


def build_frame_tree(sent, root_word, role, blocked_ids=None, visited=None):

    if not is_stanza_word(root_word):
        if root_word in ("", None):
            return {"text": "", "frame": []}
        return {"text": str(root_word), "frame": []}

    if blocked_ids is None:
        blocked_ids = set()
    if visited is None:
        visited = set()

    if root_word.id in visited:
        return {"text": frame_node_text(root_word), "frame": []}

    visited = visited | {root_word.id}

    node = {
        "text": frame_node_text(root_word),
        "frame": []
    }

    children = get_children(sent, root_word.id)

    for child in children:
        if child.id in blocked_ids:
            continue
        if child.id in visited:
            continue
        if not should_keep_child(child, role):
            continue

        child_node = build_frame_tree(
            sent=sent,
            root_word=child,
            role=role,
            blocked_ids=blocked_ids,
            visited=visited
        )
        node["frame"].append(child_node)

    return node


def convert_triplet_to_frame_struct(sent, triplet):
    core_ids = {
        value.id
        for value in triplet.values()
        if is_stanza_word(value)
    }

    result = {}

    for role, value in triplet.items():
        if is_stanza_word(value):
            blocked_ids = core_ids - {value.id}
        else:
            blocked_ids = core_ids.copy()

        result[role] = build_frame_tree(
            sent=sent,
            root_word=value,
            role=role,
            blocked_ids=blocked_ids
        )

    return result

def extract_frames(triplets_result):
    sent = triplets_result["sent_obj"]
    triplets = triplets_result["triplets"]
    new_triplets = []
    for triplet in triplets:
        new_triplets.append(convert_triplet_to_frame_struct(sent, triplet))
    return {"sentence": triplets_result["sentence"], "triplets": new_triplets}


def is_text_frame_node(x):
    return isinstance(x, dict) and "text" in x and "frame" in x


def is_nested_sentence_node(x):
    return isinstance(x, dict) and "sentence" in x and "triplets" in x


def node_is_empty(node):
    if not isinstance(node, dict):
        return True
    return not node.get("text")


def count_frame_nodes(node):
    if not is_text_frame_node(node):
        return 0
    total = 1 if node.get("text") else 0
    for child in node.get("frame", []):
        total += count_frame_nodes(child)
    return total


def role_score(triplet, role):
    node = triplet.get(role, {"text": "", "frame": []})
    if node_is_empty(node):
        return -1
    return count_frame_nodes(node)


def choose_anchor_triplet(triplets, preferred_role):
    best = None
    best_score = -1

    for triplet in triplets:
        score = role_score(triplet, preferred_role)
        if score > best_score:
            best_score = score
            best = triplet

    return best

def collect_texts(node):
    texts = set()
    if not is_text_frame_node(node):
        return texts

    if node.get("text"):
        texts.add(node["text"])

    for child in node.get("frame", []):
        texts |= collect_texts(child)

    return texts


def append_unique_frame_nodes(base_node, extra_nodes):
    seen = collect_texts(base_node)

    for node in extra_nodes:
        if not is_text_frame_node(node):
            continue
        if not node.get("text"):
            continue
        if node["text"] in seen:
            continue

        base_node["frame"].append(node)
        seen |= collect_texts(node)

    return base_node

def normalize_slot(slot, slot_role):
    if is_text_frame_node(slot):
        return slot
    if not is_nested_sentence_node(slot):
        return {"text": "", "frame": []}

    inner_triplets = slot.get("triplets", [])
    if not inner_triplets:
        return {"text": "", "frame": []}

    all_structurally_empty = all(
        node_is_empty(t.get("predicate", {"text": "", "frame": []})) and
        node_is_empty(t.get("object", {"text": "", "frame": []}))
        for t in inner_triplets
    )
    if all_structurally_empty:
        if slot_role in ("subject", "predicate", "object"):
            anchor = inner_triplets[0]
            main_node = anchor.get("subject", {"text": "", "frame": []})
            if not node_is_empty(main_node):
                return {
                    "text": main_node["text"],
                    "frame": list(main_node.get("frame", []))
                }
        sentence_text = slot.get("sentence", "")
        if sentence_text:
            return {"text": sentence_text, "frame": []}
        return {"text": "", "frame": []}

    anchor = choose_anchor_triplet(inner_triplets, slot_role)
    if anchor is None:
        return {"text": "", "frame": []}
    main_node = anchor.get(slot_role, {"text": "", "frame": []})
    if node_is_empty(main_node):
        return {"text": "", "frame": []}
    result = {
        "text": main_node["text"],
        "frame": list(main_node.get("frame", []))
    }
    for triplet in inner_triplets:
        if triplet is anchor:
            continue
        candidate = triplet.get(slot_role, {"text": "", "frame": []})
        if not node_is_empty(candidate):
            append_unique_frame_nodes(result, [candidate])
    return result

def normalize_outer_triplet(triplet):
    return {
        "subject": normalize_slot(triplet["subject"], "subject"),
        "predicate": normalize_slot(triplet["predicate"], "predicate"),
        "object": normalize_slot(triplet["object"], "object"),
    }


def process_sentence(sentence, mode='separate'):
    triplets_result = extract_one_sentence(sentence, include_full_words=True, mode=mode)
    triplets_and_frames = extract_frames(triplets_result)
    
    return triplets_and_frames

def process_triplets(triplets, mode='separate'):
    """
    Либо mode='concat' если объединять все предложения в один текст
    """
    res = {"triplets": []}
    if mode == 'separate':
        for triplet in triplets["triplets"]:
            new_triplet = dict()
            for role, value in triplet.items():
                if len(list(value["text"].split())) >= 2:
                    frames = process_sentence(value["text"], mode=mode)
                    # normalized = {
                    #     "triplets": [normalize_outer_triplet(t) for t in frames["triplets"]]
                    # }
                    new_triplet[role] = frames
                else:
                    new_triplet[role] = {"text": value["text"], "frame": value.get("frame", [])}
            res["triplets"].append(new_triplet)
        res = {"triplets":[normalize_outer_triplet(t) for t in res["triplets"]]}
    else:
        for triplet in triplets["triplets"]:
            triplets = [process_sentence(sent["text"])["triplets"] for sent in triplet.values()]
            res["triplets"].extend(triplets[0])
            res["triplets"].extend(triplets[1])
            res["triplets"].extend(triplets[2])
    res_cleaned = {"triplets": []}
    for triplet in res["triplets"]:
        if triplet["predicate"]["text"] == "":
            continue
        res_cleaned["triplets"].append(triplet)
    return res_cleaned

