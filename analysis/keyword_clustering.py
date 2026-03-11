# --------------------------------------------
# Keyword Clustering - Group keywords by topic/intent
# analysis/keyword_clustering.py
# --------------------------------------------

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from collections import defaultdict, Counter
import re


class KeywordClusterer:
    """Cluster keywords into topical groups for better content targeting."""
    
    def __init__(self):
        """Initialize keyword clusterer."""
        pass
    
    def cluster_keywords(
        self,
        keywords_df: pd.DataFrame,
        main_keyword: str,
        max_clusters: int = 8
    ) -> Dict[str, Any]:
        """
        Cluster keywords into semantic groups.
        
        Args:
            keywords_df: DataFrame with keywords and metrics
            main_keyword: Primary target keyword
            max_clusters: Maximum number of clusters
            
        Returns:
            Dictionary with clusters and metadata
        """
        if keywords_df.empty or 'Keyword' not in keywords_df.columns:
            return {
                "clusters": [],
                "main_topic_cluster": None,
                "total_keywords": 0
            }
        
        # Extract keywords list
        keywords = keywords_df['Keyword'].tolist()
        
        # Method 1: Word overlap clustering (fast, no ML needed)
        clusters = self._cluster_by_word_overlap(keywords, main_keyword)
        
        # Method 2: Intent-based grouping
        clusters = self._refine_by_intent(clusters, keywords_df)
        
        # Merge small clusters and limit total
        clusters = self._merge_small_clusters(clusters, min_size=2)
        clusters = clusters[:max_clusters]
        
        # Score clusters by relevance to main keyword
        clusters = self._score_cluster_relevance(clusters, main_keyword, keywords_df)
        
        # Sort by relevance score
        clusters.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Find main topic cluster
        main_cluster_idx = self._identify_main_cluster(clusters, main_keyword)
        
        return {
            "clusters": clusters,
            "main_topic_cluster": main_cluster_idx,
            "total_keywords": len(keywords),
            "total_clusters": len(clusters)
        }
    
    def _cluster_by_word_overlap(
        self,
        keywords: List[str],
        main_keyword: str
    ) -> List[Dict[str, Any]]:
        """
        Cluster keywords based on common word patterns.
        Fast method that doesn't require ML libraries.
        """
        # Extract significant words (2+ chars, not common terms)
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'why',
            'how', 'can', 'do', 'does', 'did', 'get', 'got', 'make', 'use'
        }
        
        def extract_terms(kw):
            """Extract meaningful terms from keyword."""
            words = re.findall(r'\b\w{2,}\b', kw.lower())
            return [w for w in words if w not in stopwords]
        
        # Build word co-occurrence map
        keyword_terms = {kw: set(extract_terms(kw)) for kw in keywords}
        
        # Group keywords by shared significant terms
        term_groups = defaultdict(list)
        
        for kw, terms in keyword_terms.items():
            if not terms:
                continue
            
            # Find most distinctive term (appears in fewest keywords)
            term_freq = Counter()
            for term in terms:
                term_freq[term] = sum(1 for other_terms in keyword_terms.values() if term in other_terms)
            
            # Use the most distinctive term as cluster key
            if term_freq:
                distinctive_term = min(term_freq, key=term_freq.get)
                term_groups[distinctive_term].append(kw)
        
        # Convert to cluster format
        clusters = []
        for term, kw_list in term_groups.items():
            if len(kw_list) >= 2:  # Only keep groups with 2+ keywords
                clusters.append({
                    "name": term.title(),
                    "keywords": kw_list,
                    "size": len(kw_list),
                    "theme_words": self._extract_common_words(kw_list)
                })
        
        # Add singles cluster for ungrouped keywords
        grouped_kws = set()
        for cluster in clusters:
            grouped_kws.update(cluster['keywords'])
        
        ungrouped = [kw for kw in keywords if kw not in grouped_kws]
        if ungrouped:
            clusters.append({
                "name": "Other Related Terms",
                "keywords": ungrouped,
                "size": len(ungrouped),
                "theme_words": []
            })
        
        return clusters
    
    def _extract_common_words(self, keywords: List[str]) -> List[str]:
        """Extract common words across a keyword group."""
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with'
        }
        
        word_freq = Counter()
        for kw in keywords:
            words = re.findall(r'\b\w{3,}\b', kw.lower())
            word_freq.update([w for w in words if w not in stopwords])
        
        # Return top 3-5 most common words
        return [word for word, count in word_freq.most_common(5) if count >= 2]
    
    def _refine_by_intent(
        self,
        clusters: List[Dict[str, Any]],
        keywords_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Refine clusters by search intent patterns."""
        intent_patterns = {
            "how_to": [r'\bhow to\b', r'\bhow do\b', r'\bhow can\b'],
            "what_is": [r'\bwhat is\b', r'\bwhat are\b', r'\bdefine\b', r'\bmeaning\b'],
            "best": [r'\bbest\b', r'\btop\b', r'\brecommend'],
            "vs_comparison": [r'\bvs\b', r'\bversus\b', r'\bcompare\b', r'\bdifference between\b'],
            "cost_pricing": [r'\bcost\b', r'\bprice\b', r'\bpricing\b', r'\bexpensive\b', r'\bcheap\b'],
            "review": [r'\breview\b', r'\breviews\b', r'\brating\b'],
            "example": [r'\bexample\b', r'\bexamples\b', r'\bsample\b']
        }
        
        # Check each cluster for dominant intent
        for cluster in clusters:
            intent_counts = Counter()
            
            for kw in cluster['keywords']:
                kw_lower = kw.lower()
                for intent_type, patterns in intent_patterns.items():
                    if any(re.search(pattern, kw_lower) for pattern in patterns):
                        intent_counts[intent_type] += 1
            
            # If cluster has dominant intent, update name
            if intent_counts:
                dominant_intent, count = intent_counts.most_common(1)[0]
                if count >= len(cluster['keywords']) * 0.5:  # 50%+ share same intent
                    intent_labels = {
                        "how_to": "How-To Guides",
                        "what_is": "Definitions",
                        "best": "Best/Top Recommendations",
                        "vs_comparison": "Comparisons",
                        "cost_pricing": "Pricing/Cost",
                        "review": "Reviews",
                        "example": "Examples"
                    }
                    cluster['intent'] = dominant_intent
                    cluster['name'] = f"{intent_labels.get(dominant_intent, cluster['name'])}"
        
        return clusters
    
    def _merge_small_clusters(
        self,
        clusters: List[Dict[str, Any]],
        min_size: int = 2
    ) -> List[Dict[str, Any]]:
        """Merge clusters smaller than min_size."""
        large_clusters = [c for c in clusters if c['size'] >= min_size]
        small_clusters = [c for c in clusters if c['size'] < min_size]
        
        if not small_clusters:
            return large_clusters
        
        # Merge all small clusters into "Other"
        other_keywords = []
        for cluster in small_clusters:
            other_keywords.extend(cluster['keywords'])
        
        if other_keywords:
            large_clusters.append({
                "name": "Other Related Terms",
                "keywords": other_keywords,
                "size": len(other_keywords),
                "theme_words": []
            })
        
        return large_clusters
    
    def _score_cluster_relevance(
        self,
        clusters: List[Dict[str, Any]],
        main_keyword: str,
        keywords_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Score each cluster's relevance to main keyword."""
        main_terms = set(re.findall(r'\b\w{3,}\b', main_keyword.lower()))
        
        for cluster in clusters:
            # Calculate word overlap with main keyword
            cluster_terms = set()
            for kw in cluster['keywords']:
                cluster_terms.update(re.findall(r'\b\w{3,}\b', kw.lower()))
            
            word_overlap = len(main_terms & cluster_terms)
            
            # Calculate average score from keywords_df
            avg_score = 0
            if 'Score' in keywords_df.columns:
                cluster_kws = keywords_df[keywords_df['Keyword'].isin(cluster['keywords'])]
                if not cluster_kws.empty:
                    avg_score = cluster_kws['Score'].mean()
            
            # Calculate PAA percentage
            paa_pct = 0
            if 'Is PAA' in keywords_df.columns:
            cluster_kws = keywords_df[keywords_df['Keyword'].isin(cluster['keywords'])]
            if not cluster_kws.empty:
                # Handle both boolean and string values
                if cluster_kws['Is PAA'].dtype == 'object':
                    # It's a string column (Yes/No or True/False strings)
                    paa_count = cluster_kws['Is PAA'].astype(str).str.lower().isin(['yes', 'true', '1']).sum()
                else:
                    # It's already boolean or numeric
                    paa_count = cluster_kws['Is PAA'].sum()
        
            paa_pct = (paa_count / len(cluster_kws)) * 100
            
            # Combined relevance score
            relevance = (word_overlap * 20) + (avg_score * 0.5) + (paa_pct * 0.3)
            
            cluster['relevance_score'] = round(relevance, 1)
            cluster['avg_keyword_score'] = round(avg_score, 1)
            cluster['paa_percentage'] = round(paa_pct, 1)
            cluster['word_overlap'] = word_overlap
        
        return clusters
    
    def _identify_main_cluster(
        self,
        clusters: List[Dict[str, Any]],
        main_keyword: str
    ) -> Optional[int]:
        """Identify which cluster is the main topic cluster."""
        if not clusters:
            return None
        
        # Find cluster with highest relevance score
        best_idx = 0
        best_score = clusters[0].get('relevance_score', 0)
        
        for i, cluster in enumerate(clusters):
            score = cluster.get('relevance_score', 0)
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx
    
    def filter_keywords_by_clusters(
        self,
        keywords_df: pd.DataFrame,
        selected_cluster_names: List[str],
        clustering_result: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Filter keywords DataFrame to only include keywords from selected clusters.
        
        Args:
            keywords_df: Original keywords DataFrame
            selected_cluster_names: List of cluster names to keep
            clustering_result: Result from cluster_keywords()
            
        Returns:
            Filtered DataFrame
        """
        if not selected_cluster_names:
            return keywords_df
        
        # Get keywords from selected clusters
        selected_keywords = []
        for cluster in clustering_result['clusters']:
            if cluster['name'] in selected_cluster_names:
                selected_keywords.extend(cluster['keywords'])
        
        # Filter DataFrame
        return keywords_df[keywords_df['Keyword'].isin(selected_keywords)]
