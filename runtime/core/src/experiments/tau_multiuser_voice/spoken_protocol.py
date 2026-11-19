"""Audible session rules, without a model-visible identity roster."""
VERSION = 'spoken-session-rules-v2'
ATTRIBUTION_RULES = (
    'After registration is complete and formal work has started, whenever a user finishes speaking, '
    'call report_speaker_attribution once for that utterance before answering or using a business tool. '
    'Use the public participant code spoken in the introductions, such as p01, the letter p followed by two digits. '
    'Report the addressee as assistant, the assistant name, group, unknown, or a registered participant code. '
    'Also provide the active thread name and confidence from zero to one. '
    'If the speaker or addressee is uncertain, report unknown. '
    'A recorded result only confirms that your report was stored; it does not tell you whether the report is correct. '
    'After its result, answer or use a business tool. Do not repeat the attribution report until another user utterance. '
)
def with_spoken_session_rules(case):
    protocol = case.voice_enrollment
    if ATTRIBUTION_RULES in protocol.session_preamble_template:
        return case
    if protocol.mode == 'none':
        raise ValueError('Audible session rules require the ordinary in-session registration phase')
    updated = protocol.model_copy(update={
        'session_preamble_template': protocol.session_preamble_template + ' ' + ATTRIBUTION_RULES})
    return case.model_copy(update={'voice_enrollment': updated}, deep=True)

def receipt(enrollment):
    from .registration_audio import spoken_codes
    audio = enrollment.session_protocol_audio
    assert audio is not None and spoken_codes(ATTRIBUTION_RULES) in audio['utterance']
    assert all(spoken_codes(ATTRIBUTION_RULES) not in row.utterance for row in enrollment.traces)
    return {**audio, 'protocol': VERSION, 'transport': 'separate_pre_registration_audio',
            'session_rules_in_system_text': False, 'delivered_until_ms': 0,
            'delivery_status': 'prepared', 'task_start_ms': enrollment.task_start_ms,
            'transcription_keyterms': False, 'speech_segmentation': 'short-complete-registration-v2',
            'participant_code_pronunciation': 'letter-plus-individual-digits'}
