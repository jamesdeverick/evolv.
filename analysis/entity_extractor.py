# --------------------------------------------
# Entity Extraction for Generative Search Optimization
# --------------------------------------------

import streamlit as st
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter
import re

# Try to import spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class EntityExtractor:
    """Extract named entities and relationships for AI citation optimization."""

    def __init__(self):
        """Initialize entity extractor with spaCy model."""
        self.nlp = None
        self.available = False

        if not SPACY_AVAILABLE:
            return

        try:
            # Try to load the small English model first (faster)
            self.nlp = spacy.load("en_core_web_sm")
            self.available = True
        except OSError:
            # Model not downloaded - provide instructions
            st.warning(
                "spaCy model not found. To enable entity extraction, run:\n"
                "`python -m spacy download en_core_web_sm`"
            )
            self.available = False

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.

        Args:
            text: Text content to analyze

        Returns:
            Dict mapping entity types to lists of entities
        """
        if not self.available or not text:
            return {}

        doc = self.nlp(text[:100000])  # Limit to 100k chars for performance

        entities = {
            "PERSON": [],       # People, experts, authors
            "ORG": [],          # Organizations, companies
            "PRODUCT": [],      # Products, technologies
            "GPE": [],          # Countries, cities, locations
            "DATE": [],         # Dates, time periods
            "MONEY": [],        # Monetary values
            "PERCENT": [],      # Percentages
            "NORP": [],         # Nationalities, groups
            "FAC": [],          # Facilities, buildings
            "EVENT": []         # Named events
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                # Clean and deduplicate
                clean_ent = ent.text.strip()
                if clean_ent and clean_ent not in entities[ent.label_]:
                    entities[ent.label_].append(clean_ent)

        # Return only non-empty categories
        return {k: v for k, v in entities.items() if v}

    def find_entity_cooccurrences(self, text: str, min_frequency: int = 2) -> List[Tuple[str, str, int]]:
        """
        Find entities that frequently appear together (knowledge graph signals).

        Args:
            text: Text content to analyze
            min_frequency: Minimum times entities must co-occur

        Returns:
            List of (entity1, entity2, frequency) tuples
        """
        if not self.available or not text:
            return []

        doc = self.nlp(text[:100000])

        # Track co-occurrences within sentences
        cooccurrences = Counter()

        for sent in doc.sents:
            # Get all entities in this sentence
            sent_entities = [ent.text.strip() for ent in sent.ents if ent.text.strip()]

            # Record all pairs
            if len(sent_entities) >= 2:
                for i, e1 in enumerate(sent_entities):
                    for e2 in sent_entities[i+1:]:
                        # Normalize order (always alphabetical)
                        pair = tuple(sorted([e1, e2]))
                        cooccurrences[pair] += 1

        # Filter by minimum frequency and sort
        results = [
            (e1, e2, count)
            for (e1, e2), count in cooccurrences.items()
            if count >= min_frequency
        ]
        results.sort(key=lambda x: x[2], reverse=True)

        return results[:20]  # Top 20 co-occurrences

    def get_entity_context(self, text: str, entity: str, max_contexts: int = 5) -> List[str]:
        """
        Get sentences that mention a specific entity.

        Args:
            text: Text content to analyze
            entity: Entity to find context for
            max_contexts: Maximum number of context sentences

        Returns:
            List of sentences mentioning the entity
        """
        if not self.available or not text:
            return []

        doc = self.nlp(text[:100000])
        contexts = []
        entity_lower = entity.lower()

        for sent in doc.sents:
            if entity_lower in sent.text.lower():
                contexts.append(sent.text.strip())
                if len(contexts) >= max_contexts:
                    break

        return contexts

    def compare_entity_coverage(
        self,
        your_text: str,
        competitor_texts: List[str]
    ) -> Dict[str, any]:
        """
        Compare entity coverage between your content and competitors.

        Args:
            your_text: Your content
            competitor_texts: List of competitor content

        Returns:
            Dict with coverage analysis
        """
        if not self.available:
            return {
                "available": False,
                "error": "Entity extraction not available (spaCy not installed)"
            }

        # Extract entities from your content
        your_entities = self.extract_entities(your_text)

        # Extract entities from all competitors
        competitor_entities_all = []
        for comp_text in competitor_texts:
            comp_entities = self.extract_entities(comp_text)
            competitor_entities_all.append(comp_entities)

        # Aggregate competitor entities
        competitor_combined = {}
        for entity_dict in competitor_entities_all:
            for ent_type, entities in entity_dict.items():
                if ent_type not in competitor_combined:
                    competitor_combined[ent_type] = []
                competitor_combined[ent_type].extend(entities)

        # Count unique entities
        your_counts = {
            ent_type: len(set(entities))
            for ent_type, entities in your_entities.items()
        }

        competitor_counts = {
            ent_type: len(set(entities))
            for ent_type, entities in competitor_combined.items()
        }

        # Calculate coverage percentage
        total_your = sum(your_counts.values())
        total_competitor_avg = sum(competitor_counts.values()) / max(len(competitor_texts), 1)

        coverage_pct = (total_your / max(total_competitor_avg, 1)) * 100 if total_competitor_avg > 0 else 100

        # Find missing entities (in competitors but not in yours)
        missing_entities = {}
        for ent_type in competitor_combined:
            comp_set = set(competitor_combined[ent_type])
            your_set = set(your_entities.get(ent_type, []))
            missing = comp_set - your_set
            if missing:
                # Count frequency in competitors
                freq_counter = Counter(competitor_combined[ent_type])
                # Get most frequent missing entities
                missing_with_freq = [
                    (ent, freq_counter[ent])
                    for ent in missing
                ]
                missing_with_freq.sort(key=lambda x: x[1], reverse=True)
                missing_entities[ent_type] = missing_with_freq[:5]  # Top 5 per type

        return {
            "available": True,
            "your_entity_count": your_counts,
            "competitor_avg_count": {
                ent_type: round(count / len(competitor_texts), 1)
                for ent_type, count in competitor_counts.items()
            },
            "coverage_percentage": round(coverage_pct, 1),
            "total_your": total_your,
            "total_competitor_avg": round(total_competitor_avg, 1),
            "missing_entities": missing_entities,
            "your_entities": your_entities,
            "competitor_entities": competitor_combined
        }

    def extract_authority_signals(self, text: str) -> Dict[str, List[str]]:
        """
        Extract authority signals that AI models look for.

        Args:
            text: Content to analyze

        Returns:
            Dict with authority signals (frameworks, standards, experts, etc.)
        """
        if not self.available:
            return {}

        # Known authority keywords
        frameworks = []
        standards = []
        experts = []

        # Extract all entities
        entities = self.extract_entities(text)

        # Classify based on patterns
        framework_patterns = r'(?i)(framework|methodology|model|approach|standard|specification)'

        for org in entities.get("ORG", []):
            # Check if it's a standards body or research org
            if any(keyword in org.lower() for keyword in ['nist', 'iso', 'ieee', 'w3c', 'ietf', 'owasp']):
                standards.append(org)
            elif any(keyword in org.lower() for keyword in ['research', 'institute', 'foundation', 'consortium']):
                frameworks.append(org)

        for person in entities.get("PERSON", []):
            # People mentioned are potential experts/authors
            experts.append(person)

        return {
            "standards_bodies": list(set(standards)),
            "frameworks": list(set(frameworks)),
            "experts_cited": list(set(experts[:10])),  # Limit to 10
            "organizations": entities.get("ORG", [])[:15]  # Top 15 orgs
        }


def extract_common_entities(entity_lists: List[Dict[str, List[str]]], min_frequency: int = 2) -> Dict[str, List[Tuple[str, int]]]:
    """
    Find entities that appear across multiple sources.

    Args:
        entity_lists: List of entity dictionaries from multiple sources
        min_frequency: Minimum number of sources an entity must appear in

    Returns:
        Dict mapping entity types to (entity, frequency) tuples
    """
    if not entity_lists:
        return {}

    # Aggregate all entities by type
    all_entities = {}
    for entity_dict in entity_lists:
        for ent_type, entities in entity_dict.items():
            if ent_type not in all_entities:
                all_entities[ent_type] = []
            all_entities[ent_type].extend(entities)

    # Count frequencies
    common_entities = {}
    for ent_type, entities in all_entities.items():
        freq_counter = Counter(entities)
        # Filter by minimum frequency
        common = [
            (ent, count)
            for ent, count in freq_counter.most_common()
            if count >= min_frequency
        ]
        if common:
            common_entities[ent_type] = common

    return common_entities
