"""Council orchestration: members translate, optionally peer-review, chairman arbitrates."""

from mt_llm.council.chairman import Chairman
from mt_llm.council.member import Member


class Council:
    """Run translation by N council members and a chairman arbiter."""

    def __init__(
        self,
        member_model_ids: list[str],
        chairman_model_id: str,
        device: str = "cpu",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        cache_dir: str | None = None,
        peer_review: bool = False,
        terms_to_chairman: bool = False,
        allow_chairman_rewrite: bool = True,
    ):
        self.members = [
            Member(
                model_id=mid,
                device=device,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
                cache_dir=cache_dir,
            )
            for mid in member_model_ids
        ]
        self.chairman = Chairman(
            model_id=chairman_model_id,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
        )
        self.peer_review = peer_review
        self.terms_to_chairman = terms_to_chairman
        self.allow_chairman_rewrite = allow_chairman_rewrite

    def load_models(self) -> None:
        """Load all member models and the chairman."""
        for m in self.members:
            m.load()
        self.chairman.load()

    def run(self, source_text: str, terms: dict[str, str] | None = None) -> dict:
        """Run the council on a single source sentence."""
        candidates = [m.translate(source_text, terms=terms) for m in self.members]

        reviews: list[str] | None = None
        if self.peer_review and len(self.members) > 1:
            reviews = []
            for i, m in enumerate(self.members):
                others = [c for j, c in enumerate(candidates) if j != i]
                reviews.append(m.review(source_text, candidates[i], others))

        chairman_terms = terms if self.terms_to_chairman else None
        final = self.chairman.arbitrate(
            source=source_text,
            candidates=candidates,
            reviews=reviews,
            terms=chairman_terms,
            allow_rewrite=self.allow_chairman_rewrite,
        )

        return {
            "source_text": source_text,
            "candidates": candidates,
            "reviews": reviews,
            "final_translation": final,
        }

    def run_batch(
        self,
        source_texts: list[str],
        terms_list: list[dict[str, str] | None] | None = None,
    ) -> list[dict]:
        if terms_list is None:
            terms_list = [None] * len(source_texts)
        return [self.run(s, terms=t) for s, t in zip(source_texts, terms_list, strict=False)]
