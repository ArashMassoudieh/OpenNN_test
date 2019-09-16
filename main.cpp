#include <iostream>
#include "opennn.h"

using namespace std;
using namespace OpenNN;

int main()
{
        DataSet data_set;

        data_set.set_data_file_name("data.dat");

        data_set.load_data();
        Matrix<double> dd = data_set.get_data();

        Variables* variables_pointer = data_set.get_variables_pointer();

        variables_pointer->set_use(0, "Input");
        variables_pointer->set_use(1, "Input");
        variables_pointer->set_use(2, "Target");

        variables_pointer->set_name(0, "x1");
        variables_pointer->set_name(1, "x2");
        variables_pointer->set_name(2, "y");

        const vector<string> inputs_name = variables_pointer->get_inputs_name_std();

        const Matrix<string> inputs_information = variables_pointer->get_inputs_information();
        const Matrix<string> targets_information = variables_pointer->get_targets_information();

        Instances* instances_pointer = data_set.get_instances_pointer();

        instances_pointer->set_training();

        const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();
        const Vector< Statistics<double> > targets_statistics = data_set.scale_targets_minimum_maximum();

        NeuralNetwork neural_network(1, 2, 1);

        neural_network.save("initial_neural_network.xml");
        neural_network.save_expression("initial_expression.txt");

        Inputs* inputs_pointer = neural_network.get_inputs_pointer();
        inputs_pointer->set_information(inputs_information);

        Outputs* outputs_pointer = neural_network.get_outputs_pointer();
        outputs_pointer->set_information(targets_information);

        neural_network.construct_scaling_layer();
        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
        scaling_layer_pointer->set_statistics(inputs_statistics);
        scaling_layer_pointer->set_scaling_methods(ScalingLayer::NoScaling);

        neural_network.construct_unscaling_layer();
        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();
        unscaling_layer_pointer->set_statistics(targets_statistics);
        unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::NoUnscaling);

        SumSquaredError sum_squared_error(&neural_network, &data_set);

        Vector<size_t> indices(0, 1, instances_pointer->get_instances_number()-1);

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::SUM_SQUARED_ERROR);
        cout << "Error method: " << training_strategy.get_sum_squared_error_pointer()->get_error_type() << endl;
        QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

        //quasi_Newton_method_pointer->set_epochs_number(1000);

        quasi_Newton_method_pointer->set_training_batch_size(11);

        quasi_Newton_method_pointer->set_display_period(10);

        quasi_Newton_method_pointer->set_maximum_epochs_number(1000);

        const TrainingStrategy::Results training_strategy_results = training_strategy.perform_training();

        neural_network.save("trained_neural_network.xml");
        //neural_network.save_expression("trained_expression.txt");


        Matrix<double> x = data_set.get_data();





        Vector<double> y = neural_network.calculate_outputs(x);

        for (int i=0; i<x.size(); i++)
            cout<<y[i]<<endl;

}
