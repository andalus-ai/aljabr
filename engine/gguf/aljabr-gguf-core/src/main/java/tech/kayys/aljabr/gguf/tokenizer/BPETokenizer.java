package tech.kayys.aljabr.gguf.loader.tokenizer;

import tech.kayys.aljabr.gguf.loader.gguf.GGUFFile;
import java.util.*;
/**
 * 
 * Core class for kayys module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class BPETokenizer {
    private final String[] pieces;
    private final Map<String, Integer> pieceToId;
    private final Map<String, Integer> merges;
    public final int bosId, eosId, unknownId;
    
    public static BPETokenizer load(GGUFFile f) {
        List<String> tokenList = f.getArray("tokenizer.ggml.tokens");
        List<String> mergeList = f.getArray("tokenizer.ggml.merges");
        
        String[] pieces = tokenList.toArray(new String[0]);
        Map<String, Integer> pieceToId = new HashMap<>();
        for (int i = 0; i < pieces.length; i++) pieceToId.put(pieces[i], i);
        
        Map<String, Integer> merges = new HashMap<>();
        for (int i = 0; i < mergeList.size(); i++) merges.put(mergeList.get(i), i);
        
        int bos = (int) f.getLong("tokenizer.ggml.bos_token_id", 1);
        int eos = (int) f.getLong("tokenizer.ggml.eos_token_id", 2);
        int unk = 0;
        
        return new BPETokenizer(pieces, pieceToId, merges, bos, eos, unk);
    }
    
    private BPETokenizer(String[] pieces, Map<String, Integer> pieceToId,
                         Map<String, Integer> merges, int bosId, int eosId, int unknownId) {
        this.pieces = pieces;
        this.pieceToId = pieceToId;
        this.merges = merges;
        this.bosId = bosId;
        this.eosId = eosId;
        this.unknownId = unknownId;
    }
    
    public int[] encode(String text, boolean addBos, boolean addEos) {
        ArrayList<Integer> ids = new ArrayList<>();
        if (addBos) ids.add(bosId);
        
        // Simple whitespace tokenization
        String[] words = text.split("(?<=\\s)|(?=\\s)");
        for (String word : words) {
            if (word.trim().isEmpty()) continue;
            encodeWord(word.replace(" ", "▁"), ids);
        }
        
        if (addEos) ids.add(eosId);
        return ids.stream().mapToInt(i -> i).toArray();
    }
    
    private void encodeWord(String word, ArrayList<Integer> out) {
        ArrayList<String> symbols = new ArrayList<>();
        for (int i = 0; i < word.length(); i++) {
            symbols.add(word.substring(i, i + 1));
        }
        
        while (symbols.size() > 1) {
            int bestPrio = Integer.MAX_VALUE;
            int bestIdx = -1;
            
            for (int i = 0; i < symbols.size() - 1; i++) {
                String merged = symbols.get(i) + " " + symbols.get(i + 1);
                Integer prio = merges.get(merged);
                if (prio != null && prio < bestPrio) {
                    bestPrio = prio;
                    bestIdx = i;
                }
            }
            
            if (bestIdx < 0) break;
            
            symbols.set(bestIdx, symbols.get(bestIdx) + symbols.get(bestIdx + 1));
            symbols.remove(bestIdx + 1);
        }
        
        for (String s : symbols) {
            Integer id = pieceToId.get(s);
            if (id != null) out.add(id);
            else out.add(unknownId);
        }
    }
    
    public String decode(int[] tokenIds) {
        StringBuilder sb = new StringBuilder();
        for (int id : tokenIds) {
            if (id >= 0 && id < pieces.length) {
                sb.append(pieces[id].replace("▁", " "));
            }
        }
        return sb.toString();
    }
}
