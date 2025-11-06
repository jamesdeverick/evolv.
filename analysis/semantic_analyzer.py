# --------------------------------------------
# Semantic Analysis for Generative Search Optimization
# --------------------------------------------

import streamlit as st
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re

# Try to import sentence-transformers and sklearn
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SemanticAnalyzer:
    """Semantic analysis using embeddings for topic coverage and clustering."""

    def __init__(self):
        """Initialize semantic analyzer with sentence transformer model."""
        self.model = None
        self.available = False

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return

        if not SKLEARN_AVAILABLE:
            return

        try:
            # Use a fast, lightweight model for semantic similarity
            # all-MiniLM-L6-v2 is 384-dim, fast, and accurate
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.available = True
        except Exception as e:
            st.warning(f"Could not load sentence transformer model: {e}")
            self.available = False

    def compute_content_coverage(
        self,
        your_content: str,
        competitor_contents: List[str],
        chunk_size: int = 200
    ) -> Dict[str, any]:
        """
        Calculate how well your content covers competitor topics semantically.

        Args:
            your_content: Your content text
            competitor_contents: List of competitor content texts
            chunk_size: Words per chunk for comparison

        Returns:
            Dict with coverage metrics
        """
        if not self.available:
            return {
                "available": False,
                "error": "Semantic analysis not available (sentence-transformers not installed)"
            }

        if not your_content or not competitor_contents:
            return {"available": True, "coverage": 0, "total_chunks": 0}

        # Split into chunks (paragraphs or fixed word count)
        your_chunks = self._split_into_chunks(your_content, chunk_size)
        competitor_chunks = []
        for comp_text in competitor_contents:
            competitor_chunks.extend(self._split_into_chunks(comp_text, chunk_size))

        if not your_chunks or not competitor_chunks:
            return {"available": True, "coverage": 0, "total_chunks": 0}

        # Generate embeddings
        your_embeds = self.model.encode(your_chunks, show_progress_bar=False)
        comp_embeds = self.model.encode(competitor_chunks, show_progress_bar=False)

        # Calculate coverage: % of competitor topics you cover
        covered_count = 0
        similarity_threshold = 0.7  # High similarity = topic is covered

        for comp_embed in comp_embeds:
            # Find max similarity to any of your chunks
            similarities = cosine_similarity([comp_embed], your_embeds)[0]
            if max(similarities) >= similarity_threshold:
                covered_count += 1

        coverage_pct = (covered_count / len(comp_embeds)) * 100

        return {
            "available": True,
            "coverage_percentage": round(coverage_pct, 1),
            "covered_topics": covered_count,
            "total_competitor_topics": len(comp_embeds),
            "your_chunks": len(your_chunks),
            "competitor_chunks": len(competitor_chunks)
        }

    def find_topic_gaps(
        self,
        your_content: str,
        competitor_contents: List[str],
        max_gaps: int = 5
    ) -> List[Dict[str, any]]:
        """
        Find specific topics competitors cover that you don't.

        Args:
            your_content: Your content
            competitor_contents: List of competitor contents
            max_gaps: Maximum number of gaps to return

        Returns:
            List of gap dictionaries with example text
        """
        if not self.available or not your_content or not competitor_contents:
            return []

        # Split into chunks
        your_chunks = self._split_into_chunks(your_content)
        competitor_chunks_with_source = []

        for i, comp_text in enumerate(competitor_contents):
            chunks = self._split_into_chunks(comp_text)
            for chunk in chunks:
                competitor_chunks_with_source.append({
                    'text': chunk,
                    'source_index': i
                })

        if not your_chunks or not competitor_chunks_with_source:
            return []

        # Generate embeddings
        your_embeds = self.model.encode(your_chunks, show_progress_bar=False)
        comp_texts = [c['text'] for c in competitor_chunks_with_source]
        comp_embeds = self.model.encode(comp_texts, show_progress_bar=False)

        # Find gaps (low similarity topics)
        gaps = []
        gap_threshold = 0.5  # Below this = significant gap

        for i, comp_embed in enumerate(comp_embeds):
            similarities = cosine_similarity([comp_embed], your_embeds)[0]
            max_similarity = max(similarities)

            if max_similarity < gap_threshold:
                gaps.append({
                    'text': competitor_chunks_with_source[i]['text'],
                    'similarity': round(float(max_similarity), 2),
                    'source': f"Competitor {competitor_chunks_with_source[i]['source_index'] + 1}",
                    'gap_score': round((1 - max_similarity) * 100, 1)  # Higher = bigger gap
                })

        # Sort by gap score (biggest gaps first)
        gaps.sort(key=lambda x: x['gap_score'], reverse=True)

        return gaps[:max_gaps]

    def cluster_keywords_semantically(
        self,
        keywords: List[str],
        max_clusters: Optional[int] = None,
        distance_threshold: float = 0.5
    ) -> Dict[str, List[str]]:
        """
        Cluster keywords by semantic meaning (replaces LLM clustering).

        Args:
            keywords: List of keywords to cluster
            max_clusters: Maximum number of clusters (None = auto)
            distance_threshold: Higher = fewer clusters (0.3-0.7 recommended)

        Returns:
            Dict mapping cluster names to keyword lists
        """
        if not self.available or not keywords:
            return {"Cluster 1": keywords}

        if len(keywords) < 3:
            return {"Main Topic": keywords}

        # Generate embeddings for all keywords
        embeds = self.model.encode(keywords, show_progress_bar=False)

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=max_clusters,
            distance_threshold=distance_threshold if max_clusters is None else None,
            linkage='average',
            metric='cosine'
        )

        labels = clustering.fit_predict(embeds)

        # Group keywords by cluster
        clusters = defaultdict(list)
        for keyword, label in zip(keywords, labels):
            clusters[label].append(keyword)

        # Name clusters based on most central keyword
        named_clusters = {}
        for label, kw_list in clusters.items():
            # Find most central keyword (highest avg similarity to others)
            if len(kw_list) == 1:
                cluster_name = kw_list[0]
            else:
                kw_indices = [keywords.index(kw) for kw in kw_list]
                kw_embeds = embeds[kw_indices]

                # Calculate centroid
                centroid = np.mean(kw_embeds, axis=0)

                # Find keyword closest to centroid
                similarities = cosine_similarity([centroid], kw_embeds)[0]
                central_idx = np.argmax(similarities)
                cluster_name = kw_list[central_idx]

            named_clusters[f"Cluster: {cluster_name}"] = kw_list

        return named_clusters

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        if not self.available or not text1 or not text2:
            return 0.0

        embeds = self.model.encode([text1, text2], show_progress_bar=False)
        similarity = cosine_similarity([embeds[0]], [embeds[1]])[0][0]

        return round(float(similarity), 3)

    def find_semantic_duplicates(
        self,
        texts: List[str],
        similarity_threshold: float = 0.85
    ) -> List[Tuple[int, int, float]]:
        """
        Find semantically duplicate or very similar texts.

        Args:
            texts: List of texts to compare
            similarity_threshold: Similarity above this = duplicate

        Returns:
            List of (index1, index2, similarity) tuples
        """
        if not self.available or len(texts) < 2:
            return []

        # Generate embeddings
        embeds = self.model.encode(texts, show_progress_bar=False)

        # Find pairs with high similarity
        duplicates = []
        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                sim = cosine_similarity([embeds[i]], [embeds[j]])[0][0]
                if sim >= similarity_threshold:
                    duplicates.append((i, j, round(float(sim), 3)))

        # Sort by similarity (highest first)
        duplicates.sort(key=lambda x: x[2], reverse=True)

        return duplicates

    def _split_into_chunks(self, text: str, chunk_size: int = 200) -> List[str]:
        """
        Split text into semantic chunks.

        Args:
            text: Text to split
            chunk_size: Target words per chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # First try to split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # If paragraphs are too long, split further
        chunks = []
        for para in paragraphs:
            words = para.split()
            if len(words) <= chunk_size:
                chunks.append(para)
            else:
                # Split long paragraph into fixed-size chunks
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i + chunk_size])
                    if chunk.strip():
                        chunks.append(chunk.strip())

        # Filter out very short chunks (less than 20 words)
        chunks = [c for c in chunks if len(c.split()) >= 20]

        return chunks

    def get_cluster_quality_score(self, keywords: List[str], clusters: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.

        Args:
            keywords: Original keywords
            clusters: Clustered keywords

        Returns:
            Dict with quality metrics
        """
        if not self.available or not keywords or not clusters:
            return {"available": False}

        # Generate embeddings
        embeds = self.model.encode(keywords, show_progress_bar=False)

        # Calculate intra-cluster similarity (cohesion)
        cluster_cohesions = []
        for cluster_name, cluster_kws in clusters.items():
            if len(cluster_kws) < 2:
                continue

            # Get embeddings for this cluster
            cluster_indices = [keywords.index(kw) for kw in cluster_kws if kw in keywords]
            if len(cluster_indices) < 2:
                continue

            cluster_embeds = embeds[cluster_indices]

            # Calculate average pairwise similarity
            similarities = []
            for i in range(len(cluster_embeds)):
                for j in range(i + 1, len(cluster_embeds)):
                    sim = cosine_similarity([cluster_embeds[i]], [cluster_embeds[j]])[0][0]
                    similarities.append(sim)

            if similarities:
                avg_cohesion = np.mean(similarities)
                cluster_cohesions.append(avg_cohesion)

        overall_cohesion = np.mean(cluster_cohesions) if cluster_cohesions else 0

        return {
            "available": True,
            "overall_cohesion": round(float(overall_cohesion), 3),
            "num_clusters": len(clusters),
            "avg_cluster_size": round(len(keywords) / max(len(clusters), 1), 1)
        }
