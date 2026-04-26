from unittest.mock import patch

from multivon_eval import EvalCase
from multivon_eval.evaluators.conversation import (
    ConversationCompleteness,
    ConversationRelevance,
    KnowledgeRetention,
    TurnConsistency,
)


def conversation_case(conversation=None, input_text="Help me plan my trip"):
    return EvalCase(input=input_text, conversation=conversation)


def sample_conversation():
    return [
        {"role": "user", "content": "I am flying to Paris next week and prefer trains over taxis."},
        {"role": "assistant", "content": "Understood. I will focus on train options in Paris."},
        {"role": "user", "content": "Please also remember that I want a museum itinerary."},
    ]


class TestConversationRelevance:
    @patch("multivon_eval.evaluators.conversation._qag_eval", return_value=(1.0, ["relevant"]))
    def test_pass_case(self, _qag_eval):
        result = ConversationRelevance().evaluate(
            conversation_case(sample_conversation()),
            "You can take the RER train from the airport and visit the Louvre first.",
        )
        assert result.passed

    @patch("multivon_eval.evaluators.conversation._qag_eval", return_value=(0.25, ["off topic"]))
    def test_fail_case(self, _qag_eval):
        result = ConversationRelevance().evaluate(
            conversation_case(sample_conversation()),
            "Here is a brownie recipe.",
        )
        assert not result.passed

    def test_edge_case_no_conversation(self):
        result = ConversationRelevance().evaluate(conversation_case(None), "Anything")
        assert not result.passed
        assert "No conversation provided" in result.reason


class TestKnowledgeRetention:
    @patch("multivon_eval.evaluators.conversation._qag_eval", return_value=(1.0, ["retained facts"]))
    def test_pass_case(self, _qag_eval):
        result = KnowledgeRetention().evaluate(
            conversation_case(sample_conversation()),
            "Since you prefer trains, take the RER and start with the Louvre.",
        )
        assert result.passed

    @patch("multivon_eval.evaluators.conversation._qag_eval", return_value=(0.33, ["forgot preference"]))
    def test_fail_case(self, _qag_eval):
        result = KnowledgeRetention().evaluate(
            conversation_case(sample_conversation()),
            "Take a taxi and skip museums.",
        )
        assert not result.passed

    def test_edge_case_no_user_turns(self):
        case = conversation_case([{"role": "assistant", "content": "Hello"}])
        result = KnowledgeRetention().evaluate(case, "Hello again")
        assert result.passed
        assert result.score == 1.0
        assert "No user turns" in result.reason

    def test_edge_case_no_conversation(self):
        result = KnowledgeRetention().evaluate(conversation_case(None), "Anything")
        assert not result.passed


class TestConversationCompleteness:
    @patch("multivon_eval.evaluators.conversation._qag_eval", return_value=(1.0, ["complete"]))
    def test_pass_case(self, _qag_eval):
        result = ConversationCompleteness().evaluate(
            conversation_case(sample_conversation()),
            "Take the RER, then visit the Louvre and Musee d'Orsay for your museum itinerary.",
        )
        assert result.passed

    @patch("multivon_eval.evaluators.conversation._qag_eval", return_value=(0.25, ["unfinished"]))
    def test_fail_case(self, _qag_eval):
        result = ConversationCompleteness().evaluate(
            conversation_case(sample_conversation()),
            "I can help with that later.",
        )
        assert not result.passed

    def test_edge_case_no_conversation(self):
        result = ConversationCompleteness().evaluate(conversation_case(None), "Anything")
        assert not result.passed


class TestTurnConsistency:
    @patch("multivon_eval.evaluators.conversation._qag_eval", return_value=(1.0, ["consistent"]))
    def test_pass_case(self, _qag_eval):
        result = TurnConsistency().evaluate(
            conversation_case(sample_conversation()),
            "As mentioned earlier, train travel fits your preference best.",
        )
        assert result.passed

    @patch("multivon_eval.evaluators.conversation._qag_eval", return_value=(0.0, ["contradiction"]))
    def test_fail_case(self, _qag_eval):
        result = TurnConsistency().evaluate(
            conversation_case(sample_conversation()),
            "Actually I never said anything about trains.",
        )
        assert not result.passed

    def test_edge_case_no_conversation(self):
        result = TurnConsistency().evaluate(conversation_case(None), "Anything")
        assert not result.passed
