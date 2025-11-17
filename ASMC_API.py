#!/usr/bin/env python3
"""
🧠 ADVANCED SEMANTIC MEMORY CLUSTERING - UNIFIED API 🧠

CREATORS:
- Sean Murphy (Human Inventor & System Architect)
- Claude AI Models (AI Co-Inventor & Implementation Partner)

A complete two-layer semantic memory system combining:
- SpatialValienceToCoords: 9D semantic coordinate generation with NLTK SentiWordNet
- Short-Term Memory: Fast RAM-based storage with semantic clustering
- Long-Term Memory: Persistent LMDB storage with spatial linking

FEATURES:
- Two-layer context retrieval (immediate + depth)
- 117k word sentiment analysis via SentiWordNet
- 9D spatial semantic clustering
- Automatic STM → LTM promotion
- Zero-shot semantic understanding

License: MIT
Copyright (c) 2024 Sean Murphy
"""

import os
import sys

# Add submodule paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'SpatialShortTermContextAIMemory'))

from STM_API import create_stm_api
from .SpatialComprehensionMap import create_scm

class AdvancedSemanticMemory:
    """
    🧠 ADVANCED SEMANTIC MEMORY CLUSTERING
    
    Unified interface for two-layer semantic memory system
    """
    
    def __init__(self, max_stm_entries: int = 50, ltm_db_path: str = "ASMC_memory.lmdb", verbose: bool = False, enable_scm: bool = True):
        """
        Initialize the Advanced Semantic Memory Clustering system
        
        Args:
            max_stm_entries: Maximum short-term memory entries (default: 50)
            ltm_db_path: Path to long-term memory database (default: ASMC_memory.lmdb)
            verbose: Enable detailed logging (default: False)
            enable_scm: Enable Spatial Comprehension Map integration (default: True)
        """
        self.max_stm_entries = max_stm_entries
        self.ltm_db_path = ltm_db_path
        self.verbose = verbose
        self.enable_scm = enable_scm
        
        # Initialize STM (which handles SVC and LTM internally)
        self._stm_api = create_stm_api(
            max_entries=max_stm_entries,
            save_interval=30,
            data_directory="./asmc_stm_data",
            ltm_db_path=ltm_db_path,
            verbose=verbose
        )
        
        # Initialize SCM (Spatial Comprehension Map)
        self.scm = None
        if enable_scm:
            self.scm = create_scm(db_path="./scm_data/scm.lmdb", verbose=verbose)
        
        if verbose:
            print("🧠 Advanced Semantic Memory Clustering initialized!")
            print(f"   STM Capacity: {max_stm_entries} entries")
            print(f"   LTM Database: {ltm_db_path}")
            print(f"   Features: Two-layer retrieval, 9D clustering, 117k word sentiment")
            if enable_scm:
                print(f"   🗺️ Spatial Comprehension Map: ENABLED")
    
    def add_experience(self, situation: str, response: str, 
                      thought: str = "", objective: str = "", action: str = "", result: str = "",
                      spatial_anchor: dict = None, metadata: dict = None):
        """
        Add an experience to memory (stores in STM, auto-promotes to LTM)
        
        Args:
            situation: The situation/context (e.g., sensor data, user input)
            response: The response/thought (e.g., LLM output, action taken)
            thought: Separated thought output from LLM
            objective: Separated objective output from LLM
            action: Action taken by agent
            result: Result of action
            spatial_anchor: Optional spatial context (dict with structure_type, cluster_id, coordinates, entities)
            metadata: Optional metadata dictionary
            
        Returns:
            Dict: Storage result with coordinate information
        """
        # Store in STM (9D semantic clustering)
        result = self._stm_api.add_conversation(
            user_message=situation,
            ai_response=response,
            thought=thought,
            objective=objective,
            action=action,
            result=result,
            metadata=metadata
        )
        
        coord_key = result.get('coordinate_key')
        
        # SCM Integration: If spatial anchor provided, link memory to location
        if self.scm and spatial_anchor and coord_key:
            try:
                # Extract spatial info
                structure_type = spatial_anchor.get('structure_type', 'spatial')
                cluster_id = spatial_anchor.get('cluster_id')
                coordinates = spatial_anchor.get('coordinates', {})
                location_type = spatial_anchor.get('context_metadata', {}).get('location_type', '')
                entities = spatial_anchor.get('entities', [])
                neighbors = spatial_anchor.get('neighbors', {})
                
                if not cluster_id:
                    return result  # Skip if no cluster specified
                
                # Ensure cluster exists
                if not self.scm.cluster_exists(cluster_id):
                    self.scm.create_cluster(
                        cluster_id=cluster_id,
                        cluster_type=structure_type,
                        description=spatial_anchor.get('context_metadata', {}).get('description', '')
                    )
                
                # Create/update node with physical structure
                node_key = self.scm.create_or_update_node(
                    structure_type=structure_type,
                    cluster_id=cluster_id,
                    coordinates=coordinates,
                    location_type=location_type,
                    entities=entities,
                    neighbors=neighbors
                )
                
                # Record visit
                self.scm.visit_node(structure_type, cluster_id, coordinates)
                
                # Extract valence from ASMC sentiment
                valence = self._extract_valence_from_stm(coord_key)
                
                # Link STM memory to SCM node
                self.scm.link_stm_memory(node_key, coord_key, valence)
                
                # Add SCM info to result
                result['scm_node_key'] = node_key
                result['scm_valence'] = valence
                
            except Exception as e:
                if self.verbose:
                    print(f"   [ASMC] Warning: SCM linking failed: {e}")
        
        return result
    
    def get_context(self, query: str, layer1_count: int = 6, layer2_count: int = 6):
        """
        Get two-layer context for a query
        
        Layer 1 (Immediate): Recent conversations + STM relevant matches
        Layer 2 (Depth): LTM semantic associations + spatial neighbors
        
        Args:
            query: Query text to build context for
            layer1_count: Items from Layer 1 (split: recent + relevant)
            layer2_count: Items from Layer 2 (split: semantic + neighbors)
            
        Returns:
            Dict with layer1_immediate and layer2_depth context
        """
        # Layer 1 split: 50/50 between recent and relevant
        recent_count = layer1_count // 2
        relevant_count = layer1_count - recent_count
        
        return self._stm_api.get_context(
            user_input=query,
            recent_count=recent_count,
            relevant_count=relevant_count
        )
    
    def get_statistics(self):
        """Get comprehensive system statistics"""
        stats = self._stm_api.get_statistics()
        
        # Add SCM statistics if enabled
        if self.scm:
            scm_stats = self.scm.get_statistics()
            stats['scm'] = scm_stats
        
        return stats
    
    def _extract_valence_from_stm(self, coord_key: str) -> float:
        """
        Extract emotional significance from STM entry using ASMC's sentiment analysis
        
        Reuses existing NLTK SentiWordNet analysis from coordinate generation.
        
        Args:
            coord_key: 9D semantic coordinate key
            
        Returns:
            float: Valence (-1.0 to +1.0)
        """
        try:
            # Get STM entry
            stm_entry = self._stm_api._stm.stm_entries.get(coord_key)
            if not stm_entry:
                return 0.0
            
            # Check if coordinate result has fingerprint with sentiment
            # (ASMC already computed this during coordinate generation)
            coord_result = stm_entry.get('coord_result', {})
            
            # Try to extract from fingerprint (if available)
            if 'fingerprint' in coord_result:
                fingerprint = coord_result['fingerprint']
                if hasattr(fingerprint, 'semantic_features'):
                    sentiment = fingerprint.semantic_features.get('sentiment', {})
                    if sentiment:
                        pos_score = sentiment.get('positive', 0.0)
                        neg_score = sentiment.get('negative', 0.0)
                        
                        # Convert to -1 to +1 scale
                        if pos_score + neg_score > 0:
                            valence = (pos_score - neg_score) / (pos_score + neg_score)
                            return valence
            
            # Fallback: Analyze response text directly (simpler heuristic)
            response = stm_entry.get('ai_response', '')
            return self._simple_valence_extraction(response)
            
        except Exception as e:
            if self.verbose:
                print(f"   [ASMC] Warning: Valence extraction failed: {e}")
            return 0.0
    
    def _simple_valence_extraction(self, text: str) -> float:
        """
        Simple valence extraction using keyword matching
        (Fallback if ASMC sentiment not available)
        """
        text_lower = text.lower()
        
        positive_keywords = [
            'good', 'great', 'excellent', 'found', 'discovered', 'satisfied',
            'happy', 'like', 'love', 'beautiful', 'amazing', 'successful',
            'comfortable', 'safe', 'peaceful', 'enjoy', 'interesting'
        ]
        
        negative_keywords = [
            'bad', 'terrible', 'failed', 'stuck', 'lost', 'confused',
            'sad', 'hate', 'scary', 'frustrating', 'painful', 'unable',
            'trapped', 'starving', 'dangerous', 'avoid'
        ]
        
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total = pos_count + neg_count
        if total > 0:
            return (pos_count - neg_count) / total
        
        return 0.0
    
    def get_spatial_context(self, structure_type: str, cluster_id: str, coordinates: dict,
                           radius: int = 0, include_ltm: bool = True, max_memories: int = 5):
        """
        Get integrated spatial + semantic context for a location
        
        Returns:
        - SCM node (structure, entities, neighbors, statistics)
        - Recent STM memories at this location
        - LTM patterns at this location
        - Cluster-level context
        
        Args:
            structure_type: Type of structure ('spatial', 'linear', etc.)
            cluster_id: Cluster identifier
            coordinates: Position within cluster
            radius: Include nearby nodes (not yet implemented)
            include_ltm: Include LTM patterns
            max_memories: Maximum STM memories to return
            
        Returns:
            Dict with complete spatial + semantic context
        """
        if not self.scm:
            return None
        
        # Get SCM node
        node = self.scm.get_node(structure_type, cluster_id, coordinates)
        if not node:
            return None
        
        # Get cluster info
        cluster = self.scm.get_cluster(cluster_id)
        
        # Fetch recent STM memories
        stm_memories = []
        for coord_key in node.get('stm_coord_keys', [])[-max_memories:]:
            entry = self._stm_api._stm.stm_entries.get(coord_key)
            if entry:
                stm_memories.append({
                    'coord_key': coord_key,
                    'semantic_summary': entry.get('semantic_summary', ''),
                    'timestamp': entry.get('timestamp', ''),
                    'valence': entry.get('valence', 0.0)
                })
        
        # Fetch LTM patterns (if requested)
        ltm_patterns = []
        if include_ltm:
            for ltm_ref in node.get('ltm_engram_ids', []):
                ltm_patterns.append({
                    'engram_id': ltm_ref.get('engram_id'),
                    'concept': ltm_ref.get('concept'),
                    'strength': ltm_ref.get('strength', 0.0)
                })
        
        return {
            'node': node,
            'cluster': cluster,
            'stm_memories': stm_memories,
            'ltm_patterns': ltm_patterns,
            'visit_count': node.get('visit_count', 0),
            'aggregate_valence': node.get('aggregate_valence', 0.0),
            'cluster_valence': cluster.get('aggregate_valence', 0.0) if cluster else 0.0
        }
    
    def get_spatial_context_string(self, structure_type: str, cluster_id: str, coordinates: dict,
                                   max_memories: int = 3) -> str:
        """
        Get human-readable spatial context string for LLM prompts
        
        Returns formatted string with:
        - Current location info
        - Physical objects/entities
        - Recent memories
        - Long-term patterns
        - Emotional valence
        - Available exits
        """
        context = self.get_spatial_context(structure_type, cluster_id, coordinates, max_memories=max_memories)
        if not context:
            return ""
        
        node = context['node']
        cluster = context['cluster']
        
        # Build context string
        lines = []
        lines.append("=== SPATIAL CONTEXT ===")
        lines.append(f"Location: {node.get('location_type', 'Unknown')} at {coordinates}")
        lines.append(f"Region: {cluster.get('description', cluster_id) if cluster else cluster_id}")
        lines.append(f"Visits: {node.get('visit_count', 0)} times")
        
        # Valence
        valence = node.get('aggregate_valence', 0.0)
        if valence > 0.3:
            feeling = "positive (+{:.2f})".format(valence)
        elif valence < -0.3:
            feeling = "negative ({:.2f})".format(valence)
        else:
            feeling = "neutral ({:.2f})".format(valence)
        lines.append(f"Feeling: {feeling}")
        
        # Objects
        entities = node.get('entities', [])
        if entities:
            lines.append(f"Objects here: {', '.join(entities)}")
        
        # Recent memories
        if context['stm_memories']:
            lines.append("\nRecent experiences:")
            for mem in context['stm_memories'][:3]:
                lines.append(f"  - {mem['semantic_summary'][:60]}...")
        
        # Long-term patterns
        if context['ltm_patterns']:
            lines.append("\nKnown patterns:")
            for pat in context['ltm_patterns'][:3]:
                lines.append(f"  - {pat['concept']} (strength: {pat['strength']:.2f})")
        
        # Neighbors/exits
        neighbors = node.get('neighbors', {})
        if neighbors:
            lines.append("\nAvailable exits:")
            for direction, neighbor in neighbors.items():
                if neighbor:
                    lines.append(f"  - {direction}")
        
        return "\n".join(lines)
    
    def clear_memory(self, confirm: bool = False):
        """Clear all memories (DESTRUCTIVE - requires confirm=True)"""
        result = self._stm_api.clear_memory(confirm=confirm)
        
        # Also clear SCM if enabled
        if self.scm and confirm:
            # Close and recreate SCM (clears database)
            self.scm.close()
            import os
            scm_path = "./scm_data/scm.lmdb"
            if os.path.exists(scm_path):
                try:
                    os.remove(scm_path)
                    os.remove(scm_path + "-lock")
                except:
                    pass
            self.scm = create_scm(db_path=scm_path, verbose=self.verbose)
        
        return result
    
    def shutdown(self):
        """Gracefully shutdown the memory system"""
        return self._stm_api.shutdown()
    
    def MassDataUpload(self, folder_path, file_extensions=['.txt', '.md'], chunk_size=300):
        """
        Matrix-style knowledge upload - scan folder and inject all text files as memories.
        Auto-chunks large files and uploads via existing STM pipeline (auto-promotes to LTM).
        
        Args:
            folder_path: Path to folder containing text files
            file_extensions: File types to process (default: ['.txt', '.md'])
            chunk_size: Size of text chunks in characters (default: 300)
        
        Returns:
            Dict: Upload statistics
        """
        import os
        
        print("🧠" * 30)
        print("🧠 ASMC MASS DATA UPLOAD - MATRIX MODE")
        print("🧠" * 30)
        print(f"📁 Scanning: {folder_path}")
        print(f"📄 File types: {', '.join(file_extensions)}")
        print(f"✂️ Chunk size: {chunk_size} chars")
        print("="*70 + "\n")
        
        # Find all files
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    all_files.append(os.path.join(root, file))
        
        if not all_files:
            print("❌ No files found!")
            return {'error': 'No files found', 'files_processed': 0}
        
        print(f"📚 Found {len(all_files)} files\n")
        
        total_chunks = 0
        files_processed = 0
        
        # Process each file
        for file_idx, filepath in enumerate(all_files, 1):
            filename = os.path.basename(filepath)
            print(f"📖 [{file_idx}/{len(all_files)}] Processing: {filename}")
            
            try:
                # Read file
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                if not text.strip():
                    print(f"   ⚠️ Empty file, skipping")
                    continue
                
                # Chunk text
                chunks = self._chunk_text(text, chunk_size)
                print(f"   ✂️ Created {len(chunks)} chunks")
                
                # Upload each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    self.add_experience(
                        situation=f"Knowledge from {filename}",
                        response=chunk,
                        metadata={
                            'source': 'mass_upload',
                            'filename': filename,
                            'filepath': filepath,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'is_innate': True
                        }
                    )
                    total_chunks += 1
                    
                    # Progress every 100 chunks
                    if total_chunks % 100 == 0:
                        stats = self.get_statistics()
                        promoted = stats.get('total_promoted_to_longterm', 0)
                        print(f"   📊 Progress: {total_chunks} chunks uploaded | {promoted} promoted to LTM")
                
                files_processed += 1
                print(f"   ✅ {filename} complete ({len(chunks)} chunks)\n")
                
            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}\n")
        
        # Final statistics
        final_stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("🎉 MASS UPLOAD COMPLETE!")
        print("="*70)
        print(f"📁 Files processed: {files_processed}/{len(all_files)}")
        print(f"📝 Total chunks: {total_chunks}")
        print(f"🧠 STM entries: {final_stats.get('current_entries', 0)}")
        print(f"📤 Promoted to LTM: {final_stats.get('total_promoted_to_longterm', 0)}")
        print(f"💾 Memory system ready with pre-loaded knowledge!")
        print("="*70)
        
        return {
            'success': True,
            'files_found': len(all_files),
            'files_processed': files_processed,
            'chunks_uploaded': total_chunks,
            'promoted_to_ltm': final_stats.get('total_promoted_to_longterm', 0),
            'current_stm_entries': final_stats.get('current_entries', 0)
        }
    
    def _chunk_text(self, text, chunk_size=300):
        """Split text into semantic chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        # Split on sentences
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk + ".")
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk + ".")
        
        return chunks


# Convenience factory function
def create_memory(max_entries: int = 50, db_path: str = "ASMC_memory.lmdb", verbose: bool = False):
    """
    Quick factory function to create Advanced Semantic Memory system
    
    Args:
        max_entries: Maximum STM entries
        db_path: LTM database path  
        verbose: Enable logging
        
    Returns:
        AdvancedSemanticMemory: Initialized memory system
    """
    return AdvancedSemanticMemory(
        max_stm_entries=max_entries,
        ltm_db_path=db_path,
        verbose=verbose
    )


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED SEMANTIC MEMORY CLUSTERING - Example")
    print("=" * 60)
    
    # Create memory system
    memory = create_memory(max_entries=10, verbose=True)
    
    # Add some experiences
    print("\nAdding experiences...")
    memory.add_experience(
        situation="I took cursed amulet in Room (1,1)",
        response="Lost 25 HP! Health now 75/100"
    )
    
    memory.add_experience(
        situation="I took healing potion in Room (2,2)",
        response="Gained 30 HP! Health now 100/100"
    )
    
    # Get context for new situation
    print("\nGetting context for: 'I see a cursed amulet'")
    context = memory.get_context("I see a cursed amulet", layer1_count=6, layer2_count=6)
    
    if context['success']:
        print(f"\nLayer 1 (Immediate):")
        print(f"  Recent: {len(context.get('recent_context', []))} items")
        print(f"  Relevant: {len(context.get('relevant_context', []))} items")
        
        print(f"\nLayer 2 (Depth):")
        print(f"  Semantic: {len(context.get('ltm_semantic', []))} items")
        print(f"  Neighbors: {len(context.get('ltm_neighbors', []))} items")
        
        print(f"\nTotal context: {context['total_context_entries']} items")
    
    # Shutdown
    print("\nShutting down...")
    memory.shutdown()
    print("Complete!")

