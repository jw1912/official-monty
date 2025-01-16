mod inputs;
mod loader;
mod moves;
mod preparer;
mod trainer;

use bullet::{
    nn::{
        optimiser::{AdamWOptimiser, AdamWParams, Optimiser}, Activation, ExecutionContext, Graph, NetworkBuilder, Node, Shape
    }, trainer::{
        default::loader::DataLoader,
        logger,
        save::{Layout, QuantTarget, SavedFormat},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
        NetworkTrainer, DataPreparer
    },
};

use trainer::Trainer;

const ID: &str = "policy001";
const DATA: &str = "ataxxgen001.binpack";

fn main() {
    let data_preparer = preparer::DataPreparer::new(DATA, 4096, 4);

    let size = 128;

    let (graph, node) = network(size);

    let optimiser_params = AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    };

    let mut trainer = Trainer {
        optimiser: AdamWOptimiser::new(graph, optimiser_params),
    };

    let schedule = TrainingSchedule {
        net_id: ID.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 40,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 18 },
        save_rate: 5,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
    };

    logger::clear_colours();
    println!("{}", logger::ansi("Beginning Training", "34;1"));
    schedule.display();
    settings.display();

    //trainer.load_from_checkpoint("checkpoints/policy001-40");
    //eval_random(&mut trainer, node, &data_preparer);

    trainer.train_custom(
        &data_preparer,
        &Option::<preparer::DataPreparer>::None,
        &schedule,
        &settings,
        |sb, trainer, schedule, _| {
            if schedule.should_save(sb) {
                save(trainer, &format!("checkpoints/{ID}-{sb}/quantised.network"));
            }
        },
    );

    let data_preparer = preparer::DataPreparer::new(DATA, 4096, 1);
    for _ in 0..5 {
        eval_random(&mut trainer, node, &data_preparer);
    }
}

fn eval_random(trainer: &mut Trainer, node: Node, data_preparer: &preparer::DataPreparer) {
    data_preparer.loader.map_batches(0, 1, |batch| {
        println!("{}", batch[0].pos.as_fen());

        let indices: Vec<_> = batch[0].moves.iter().take(batch[0].num).map(|x| moves::map_move_to_index(x.0)).collect();

        let prepd = data_preparer.prepare(batch, 1, 1.0);
        trainer.load_batch(&prepd);
        trainer.optimiser.graph_mut().forward();
        let mut vals = trainer.optimiser.graph().get_node(node).get_dense_vals().unwrap();

        let mut max = 0.0;
        let mut total = 0.0;
        for &i in &indices {
            max = vals[i].max(max);
        }

        for &i in &indices {
            vals[i] = (vals[i] - max).exp();
            total += vals[i];
        }

        for (j, &i) in indices.iter().enumerate() {
            vals[i] /= total;
            println!("{} -> {:.2}%", batch[0].moves[j].0.uai(), 100.0 * vals[i]);
        }

        true
    });
}

fn save(trainer: &Trainer, path: &str) {
    trainer
    .save_weights_portion(
        path,
        &[
            SavedFormat::new("l0w", QuantTarget::I8(128), Layout::Normal),
            SavedFormat::new("l0b", QuantTarget::I8(128), Layout::Normal),
            SavedFormat::new("l1w", QuantTarget::I8(128), Layout::Transposed),
            SavedFormat::new("l1b", QuantTarget::I8(128), Layout::Normal),
        ],
    )
    .unwrap();
}

fn network(size: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    let inputs = builder.new_input("inputs", Shape::new(inputs::INPUT_SIZE, 1));
    let mask = builder.new_input("mask", Shape::new(moves::NUM_MOVES, 1));
    let dist = builder.new_input("dist", Shape::new(moves::MAX_MOVES, 1));

    let l0 = builder.new_affine("l0", inputs::INPUT_SIZE, size);
    let l1 = builder.new_affine("l1", size, moves::NUM_MOVES);

    let mut out = l0.forward(inputs).activate(Activation::CReLU);
    out = l1.forward(out);
    out.masked_softmax_crossentropy_loss(dist, mask);

    let o = out.node();
    (builder.build(ExecutionContext::default()), o)
}
