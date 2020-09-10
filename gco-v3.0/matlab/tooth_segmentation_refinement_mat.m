function tooth_segmentation_refinement_mat(path, sample_name, lambda, round_factor)
% lamda = 20 for teeth-gingiva; lamda=70 for interval teeth
%round_factor = 100 - 10000
    
    file_name = join([path, 'predicted_labels_', sample_name, '.mat']);
    file2_name = join([path, 'prob_labels_', sample_name, '.mat']);
    file3_name = join([path, 'neighbor_terms_', sample_name, '.mat']);
    output_name = join([path, 'refine_label_', sample_name, '.mat']);
    
    init_label_mat = load(file_name);
    init_label = init_label_mat.predicted_labels;
    init_label = init_label + 1; %change to 1-based label

    prob_mat = load(file2_name);
    prob = prob_mat.prob_labels';

    num_cells = length(init_label);
    num_labels = size(prob, 1);

    prob_cost = -log10(prob)*double(round_factor);

    %=====================================
    h = GCO_Create(num_cells, num_labels);

    GCO_SetLabeling(h, init_label)

    GCO_SetDataCost(h, prob_cost)

    smooth_cost = (1.0 - eye(num_labels));
    GCO_SetSmoothCost(h, smooth_cost);

    neighbor_terms = load(file3_name);
    neighbor_mat = neighbor_terms.neighbor_terms;
    GCO_SetNeighbors(h, neighbor_mat*double(lambda)*double(round_factor))

    GCO_Expansion(h);
    [E D S] = GCO_ComputeEnergy(h);
    refine_label = GCO_GetLabeling(h);
    GCO_Delete(h);

    save(output_name, 'refine_label');
    
end
