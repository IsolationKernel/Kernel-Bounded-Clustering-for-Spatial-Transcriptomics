function [refine_label] = refineMethod(pos, tclass,shape)
if shape == 'square'
    num_obs = 4;
elseif shape == 'hexagon'
    num_obs = 6;
else 
    disp("'Select shape='hexagon' for Visium data, 'square' for ST data.'")
    num_obs = 6;
end

refined_pred = tclass;
length = size(tclass,1);
adjacent = pdist2(pos,pos,'euclidean');

for label_index = 1:length
    dist = adjacent(label_index,:)';
    [MinkKvalue,minkindex]=mink(dist,num_obs+1);
    label_ht = java.util.Hashtable;
    minkindex = minkindex(2:num_obs+1);
    for li = 1:num_obs
       if label_ht.containsKey(refined_pred(minkindex(li)))
           label_ht.put(refined_pred(minkindex(li)), label_ht.get(refined_pred(minkindex(li)))+1);
       else
        label_ht.put(refined_pred(minkindex(li)),1);
       end
    end
    label_list = label_ht.keys;
    labelht = [];
    valueht = [];
    while( label_list.hasNext )
        word = label_list.nextElement;
        labelht = [labelht;word];
        valueht = [valueht;label_ht.get(word)];
    end
    [maxht,indexht]= max(valueht);
   
    if label_ht.containsKey(refined_pred(label_index))
        first_judge = label_ht.get(refined_pred(label_index));
    else
        first_judge = 0;
    end
    if(first_judge <(num_obs/2) && maxht>(num_obs/2))
       refined_pred(label_index) = labelht(indexht);  
    end
end
refine_label = refined_pred;

