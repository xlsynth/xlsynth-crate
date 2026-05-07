// SPDX-License-Identifier: Apache-2.0

import types.widget as shared_widget;

pub fn bump_widget(widget: shared_widget::Widget) -> shared_widget::Widget {
    shared_widget::Widget { widget_id: widget.widget_id + u8:1 }
}
