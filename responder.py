"""
responder.py
Generates a draft auto-response for a classified ticket. Purely
template-based so the project runs with zero external API keys, but the
generate_reply() function is written so it's a one-line swap to call an
LLM (OpenAI/Anthropic/etc.) instead if you want to extend it later.
"""

TEMPLATES = {
    "Technical Issue": (
        "Hi {name},\n\n"
        "Thanks for reporting this. We've logged your issue with the "
        "{category} team and assigned it priority: {priority}. "
        "Our engineers will begin investigating right away and will "
        "update you as soon as we have a fix or a workaround.\n\n"
        "In the meantime, could you confirm which browser/device you're "
        "using and whether this affects all users or just your account?\n\n"
        "Ticket ID: {ticket_id}\n"
        "— Support Team"
    ),
    "Billing": (
        "Hi {name},\n\n"
        "Thank you for reaching out about your invoice. We've forwarded "
        "this to our billing team (priority: {priority}) and they'll "
        "review the charges and respond within one business day. If this "
        "turns out to be an error, any adjustment will be reflected on "
        "your next statement.\n\n"
        "Ticket ID: {ticket_id}\n"
        "— Billing Support"
    ),
    "Account Access": (
        "Hi {name},\n\n"
        "We've received your access request (priority: {priority}). For "
        "your security, our team will verify your identity before making "
        "any account changes — you may receive a follow-up verification "
        "email shortly.\n\n"
        "Ticket ID: {ticket_id}\n"
        "— Identity & Access Team"
    ),
    "Feature Request": (
        "Hi {name},\n\n"
        "Thanks for the suggestion! We've added this to our product "
        "feedback backlog for review by the product team. We can't "
        "promise a timeline, but requests like this genuinely help us "
        "prioritize the roadmap.\n\n"
        "Ticket ID: {ticket_id}\n"
        "— Product Team"
    ),
    "General Inquiry": (
        "Hi {name},\n\n"
        "Thanks for getting in touch. A member of our support team will "
        "follow up with the information you requested shortly.\n\n"
        "Ticket ID: {ticket_id}\n"
        "— Support Team"
    ),
}


def generate_reply(name: str, category: str, priority: str, ticket_id: int) -> str:
    """Return a drafted auto-response for the given classified ticket.

    Swap this implementation for a call to an LLM API if you want fully
    generative replies later — everything else in the app stays the same.
    """
    template = TEMPLATES.get(category, TEMPLATES["General Inquiry"])
    return template.format(
        name=name or "there",
        category=category,
        priority=priority,
        ticket_id=ticket_id,
    )
