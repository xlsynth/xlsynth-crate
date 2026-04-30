// SPDX-License-Identifier: Apache-2.0

import shared_widget_types;

pub fn bump_widget(widget: shared_widget_types::Widget) -> shared_widget_types::Widget {
    shared_widget_types::Widget { widget_id: widget.widget_id + u8:1 }
}
