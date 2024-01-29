
def sampling_batch(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False, rot_bool=True, tr_bool=True, tor_bool=True):
    N = len(data_list)

    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

        # loader = DataLoader(data_list, batch_size=30) # xqup
        dataset = Datalist_to_PDBBind(data_list=data_list)
        b = 30
        loader = DataListLoader(dataset=dataset, batch_size=b, num_workers=4, shuffle=False)

        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms, device)
            
            with torch.no_grad():
                # 平移分数,旋转分数,扭转分数
                tr_score, rot_score, tor_score = model(complex_graph_batch)

            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

                rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

            if not model_args.no_torsion:
            # if not True:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                if ode:
                    tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
                torsions_per_molecule = tor_perturb.shape[0] // b
            else:
                tor_perturb = None

            batch_torsion_num = [complex_graph['ligand'].edge_mask.sum() for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())]
            start_idx = [0] + [sum(batch_torsion_num[:idx]) for idx in range(1,len(batch_torsion_num))]
            end_ix = [sum(batch_torsion_num[:idx]) for idx in range(1,len(batch_torsion_num)+1)]
            # Apply noise
            new_data_list.extend(
                
                [modify_conformer(complex_graph, tr_perturb[i:i + 1] * float(tr_bool), rot_perturb[i:i + 1].squeeze(0) * float(rot_bool),
                                          tor_perturb[start_idx[i]:end_ix[i]] if tor_bool else None)
                         for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())]
                         )
            # new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1] * float(tr_bool), rot_perturb[i:i + 1].squeeze(0) * float(rot_bool),
            #                              None)
            #             for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
        data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    with torch.no_grad():
        # 置信模型
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(confidence_complex_graph_batch, 0, 0, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence