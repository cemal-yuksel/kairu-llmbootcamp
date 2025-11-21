
import { RagChunk } from '../types';

/**
 * Computes the Cosine Similarity between two vectors.
 * Similarity = (A . B) / (||A|| * ||B||)
 * 
 * @param vecA First vector (array of numbers)
 * @param vecB Second vector (array of numbers)
 * @returns A number between -1 and 1 (1 being identical)
 */
export const cosineSimilarity = (vecA: number[], vecB: number[]): number => {
    if (vecA.length !== vecB.length) {
        console.warn("Vector dimensions do not match");
        return 0;
    }

    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        magnitudeA += vecA[i] * vecA[i];
        magnitudeB += vecB[i] * vecB[i];
    }

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    if (magnitudeA === 0 || magnitudeB === 0) return 0;

    return dotProduct / (magnitudeA * magnitudeB);
};

/**
 * Performs Semantic Search on a collection of RAG chunks.
 * 
 * @param queryEmbedding The embedding vector of the search query
 * @param chunks All available chunks from active files
 * @param topK Number of results to return (default 20)
 * @returns Sorted list of chunks by relevance with scores injected
 */
export const findMostRelevantChunks = (
    queryEmbedding: number[], 
    chunks: RagChunk[], 
    topK: number = 20
): RagChunk[] => {
    
    // 1. Filter chunks that actually have embeddings
    const validChunks = chunks.filter(c => c.embedding && c.embedding.length > 0);

    if (validChunks.length === 0) return [];

    // 2. Calculate scores
    const scoredChunks = validChunks.map(chunk => {
        return {
            chunk,
            score: cosineSimilarity(queryEmbedding, chunk.embedding!)
        };
    });

    // 3. Sort by score (Descending)
    scoredChunks.sort((a, b) => b.score - a.score);

    // 4. Return top K content with score injected
    return scoredChunks.slice(0, topK).map(sc => ({
        ...sc.chunk,
        score: sc.score // Inject the score into the chunk object
    }));
};
