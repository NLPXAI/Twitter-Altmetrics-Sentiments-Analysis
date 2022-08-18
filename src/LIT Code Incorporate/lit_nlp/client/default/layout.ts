/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Import Modules
import '../modules/span_graph_module';

import {LitModuleType} from '../core/lit_module';
import {LitComponentLayouts} from '../lib/types';
import {AnnotatedTextGoldModule, AnnotatedTextModule} from '../modules/annotated_text_module';
import {AttentionModule} from '../modules/attention_module';
import {ClassificationModule} from '../modules/classification_module';
import {ColorModule} from '../modules/color_module';
import {ConfusionMatrixModule} from '../modules/confusion_matrix_module';
import {CounterfactualExplainerModule} from '../modules/counterfactual_explainer_module';
import {DataTableModule, SimpleDataTableModule} from '../modules/data_table_module';
import {DatapointEditorModule, SimpleDatapointEditorModule} from '../modules/datapoint_editor_module';
import {EmbeddingsModule} from '../modules/embeddings_module';
import {GeneratedImageModule} from '../modules/generated_image_module';
import {GeneratedTextModule} from '../modules/generated_text_module';
import {GeneratorModule} from '../modules/generator_module';
import {LanguageModelPredictionModule} from '../modules/lm_prediction_module';
import {MetricsModule} from '../modules/metrics_module';
import {MultilabelModule} from '../modules/multilabel_module';
import {PdpModule} from '../modules/pdp_module';
import {RegressionModule} from '../modules/regression_module';
import {SalienceMapModule} from '../modules/salience_map_module';
import {ScalarModule} from '../modules/scalar_module';
import {SequenceSalienceModule} from '../modules/sequence_salience_module';
import {SliceModule} from '../modules/slice_module';
import {SpanGraphGoldModuleVertical, SpanGraphModuleVertical} from '../modules/span_graph_module';
import {TCAVModule} from '../modules/tcav_module';
import {ThresholderModule} from '../modules/thresholder_module';

// clang-format off
const MODEL_PREDS_MODULES: LitModuleType[] = [
  SpanGraphGoldModuleVertical,
  SpanGraphModuleVertical,
  ClassificationModule,
  MultilabelModule,
  RegressionModule,
  LanguageModelPredictionModule,
  GeneratedTextModule,
  AnnotatedTextGoldModule,
  AnnotatedTextModule,
  GeneratedImageModule,
];

const DEFAULT_MAIN_GROUP: LitModuleType[] = [
  DataTableModule,
  DatapointEditorModule,
  SliceModule,
  ColorModule,
];
// clang-format on

// clang-format off
/**
 * Possible layouts for LIT (component groups and settigns.)
 */
export const LAYOUTS: LitComponentLayouts = {
  /**
   * A "simple demo server" layout.
   */
  'simple':  {
    upper: {
      "Editor": [SimpleDatapointEditorModule],
      "Examples": [SimpleDataTableModule],
    },
    lower: {
      'Predictions': [ ...MODEL_PREDS_MODULES],
      'Salience': [SalienceMapModule, SequenceSalienceModule],
    },
    layoutSettings: {
      hideToolbar: true,
      mainHeight: 30,
      centerPage: true
    },
    description: 'A basic layout just containing a datapoint creator/editor, the predictions, and the data table. There are also some visual simplifications: the toolbar is hidden, and the modules are centered on the page rather than being full width.'
  },
  /**
   * A default layout for LIT Modules
   */
  'default':  {
    components : {
      'Main': [EmbeddingsModule, ...DEFAULT_MAIN_GROUP],
      'Predictions': [
        ...MODEL_PREDS_MODULES,
        ScalarModule,
        PdpModule,
      ],
      'Explanations': [
        ...MODEL_PREDS_MODULES,
        SalienceMapModule,
        SequenceSalienceModule,
        AttentionModule,
      ],
      'Metrics': [
        MetricsModule,
        ConfusionMatrixModule,
        ThresholderModule,
      ],
      'Counterfactuals': [GeneratorModule, CounterfactualExplainerModule],
      'TCAV': [
        TCAVModule,
      ],
    },
    description: "The default LIT layout, which includes the data table and data point editor, the performance and metrics, predictions, explanations, and counterfactuals."
  },
};
// clang-format on
