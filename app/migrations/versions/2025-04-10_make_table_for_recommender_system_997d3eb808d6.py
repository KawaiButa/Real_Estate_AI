# type: ignore
"""Make table for recommender system

Revision ID: 997d3eb808d6
Revises: dfa22bca0d19
Create Date: 2025-04-10 02:59:05.940463

"""

import warnings
from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from advanced_alchemy.types import EncryptedString, EncryptedText, GUID, ORA_JSONB, DateTimeUTC
from sqlalchemy import Text  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["downgrade", "upgrade", "schema_upgrades", "schema_downgrades", "data_upgrades", "data_downgrades"]

sa.GUID = GUID
sa.DateTimeUTC = DateTimeUTC
sa.ORA_JSONB = ORA_JSONB
sa.EncryptedString = EncryptedString
sa.EncryptedText = EncryptedText

# revision identifiers, used by Alembic.
revision = '997d3eb808d6'
down_revision = 'dfa22bca0d19'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with op.get_context().autocommit_block():
            schema_upgrades()
            data_upgrades()

def downgrade() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with op.get_context().autocommit_block():
            data_downgrades()
            schema_downgrades()

def schema_upgrades() -> None:
    """schema upgrade migrations go here."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user_actions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=True),
    sa.Column('property_id', sa.UUID(), nullable=True),
    sa.Column('action', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['property_id'], ['images.id'], name=op.f('fk_user_actions_property_id_images'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['images.id'], name=op.f('fk_user_actions_user_id_images'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_user_actions')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_user_actions_id')),
    sa.UniqueConstraint('property_id'),
    sa.UniqueConstraint('property_id', name=op.f('uq_user_actions_property_id')),
    sa.UniqueConstraint('user_id'),
    sa.UniqueConstraint('user_id', name=op.f('uq_user_actions_user_id'))
    )
    op.create_table('user_searchs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=True),
    sa.Column('search_query', sa.String(), nullable=True),
    sa.Column('type', sa.String(), nullable=True),
    sa.Column('min_price', sa.Numeric(precision=12, scale=2, asdecimal=False), nullable=True),
    sa.Column('max_price', sa.Numeric(precision=12, scale=2, asdecimal=False), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['images.id'], name=op.f('fk_user_searchs_user_id_images'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_user_searchs')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_user_searchs_id')),
    sa.UniqueConstraint('user_id'),
    sa.UniqueConstraint('user_id', name=op.f('uq_user_searchs_user_id'))
    )
    with op.batch_alter_table('banners', schema=None) as batch_op:
        batch_op.create_unique_constraint(batch_op.f('uq_banners_id'), ['id'])

    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('min_price', sa.Numeric(precision=12, scale=2, asdecimal=False), nullable=True))
        batch_op.add_column(sa.Column('max_price', sa.Numeric(precision=12, scale=2, asdecimal=False), nullable=True))

    # ### end Alembic commands ###

def schema_downgrades() -> None:
    """schema downgrade migrations go here."""
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('max_price')
        batch_op.drop_column('min_price')

    with op.batch_alter_table('banners', schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f('uq_banners_id'), type_='unique')

    op.create_table('spatial_ref_sys',
    sa.Column('srid', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('auth_name', sa.VARCHAR(length=256), autoincrement=False, nullable=True),
    sa.Column('auth_srid', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('srtext', sa.VARCHAR(length=2048), autoincrement=False, nullable=True),
    sa.Column('proj4text', sa.VARCHAR(length=2048), autoincrement=False, nullable=True),
    sa.CheckConstraint('srid > 0 AND srid <= 998999', name='spatial_ref_sys_srid_check'),
    sa.PrimaryKeyConstraint('srid', name='spatial_ref_sys_pkey')
    )
    op.drop_table('user_searchs')
    op.drop_table('user_actions')
    # ### end Alembic commands ###

def data_upgrades() -> None:
    """Add any optional data upgrade migrations here!"""

def data_downgrades() -> None:
    """Add any optional data downgrade migrations here!"""
